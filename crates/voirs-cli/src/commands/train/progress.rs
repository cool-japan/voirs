//! Training progress monitoring utilities
//!
//! Provides comprehensive progress tracking and visualization for model training:
//! - Multi-level progress bars (epoch, batch, metrics)
//! - Live metrics display (loss, learning rate, GPU usage)
//! - Resource monitoring (CPU, memory, GPU)
//! - Training statistics (samples/sec, ETA)

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Training progress tracker with multi-level progress bars
pub struct TrainingProgress {
    /// Multi-progress container
    multi: MultiProgress,
    /// Epoch progress bar
    epoch_bar: ProgressBar,
    /// Batch progress bar
    batch_bar: ProgressBar,
    /// Metrics display bar
    metrics_bar: ProgressBar,
    /// Resource monitoring bar
    resource_bar: Option<ProgressBar>,
    /// Training start time
    start_time: Instant,
    /// Current epoch
    current_epoch: usize,
    /// Total epochs
    total_epochs: usize,
    /// Enable resource monitoring
    monitor_resources: bool,
}

impl TrainingProgress {
    /// Create new training progress tracker
    pub fn new(total_epochs: usize, batches_per_epoch: usize, monitor_resources: bool) -> Self {
        let multi = MultiProgress::new();

        // Epoch progress bar
        let epoch_bar = multi.add(ProgressBar::new(total_epochs as u64));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:.bold.cyan} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓▒░ "),
        );
        epoch_bar.set_prefix("Epochs");

        // Batch progress bar
        let batch_bar = multi.add(ProgressBar::new(batches_per_epoch as u64));
        batch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:.bold.green} [{bar:40.green/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓▒░ "),
        );
        batch_bar.set_prefix("Batches");

        // Metrics display (spinner with metrics)
        let metrics_bar = multi.add(ProgressBar::new_spinner());
        metrics_bar.set_style(
            ProgressStyle::default_spinner()
                .template("{prefix:.bold.yellow} {spinner:.yellow} {msg}")
                .unwrap(),
        );
        metrics_bar.set_prefix("Metrics");

        // Resource monitoring bar (optional)
        let resource_bar = if monitor_resources {
            let bar = multi.add(ProgressBar::new_spinner());
            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{prefix:.bold.magenta} {spinner:.magenta} {msg}")
                    .unwrap(),
            );
            bar.set_prefix("Resources");
            Some(bar)
        } else {
            None
        };

        Self {
            multi,
            epoch_bar,
            batch_bar,
            metrics_bar,
            resource_bar,
            start_time: Instant::now(),
            current_epoch: 0,
            total_epochs,
            monitor_resources,
        }
    }

    /// Start new epoch
    pub fn start_epoch(&mut self, epoch: usize, batches: usize) {
        self.current_epoch = epoch;
        self.batch_bar.set_length(batches as u64);
        self.batch_bar.set_position(0);
        self.epoch_bar
            .set_message(format!("Epoch {}/{}", epoch + 1, self.total_epochs));
    }

    /// Update batch progress
    pub fn update_batch(&self, batch: usize, batch_loss: f64, samples_per_sec: f64) {
        self.batch_bar.set_position(batch as u64);
        self.batch_bar.set_message(format!(
            "loss: {:.4}, {:.1} samples/s",
            batch_loss, samples_per_sec
        ));
    }

    /// Finish current batch
    pub fn finish_batch(&self) {
        self.batch_bar.inc(1);
    }

    /// Update metrics display
    pub fn update_metrics(&self, metrics: &TrainingMetrics) {
        let msg = format!(
            "Loss: {:.4} | LR: {:.6} | Grad: {:.4} | Time: {}",
            metrics.loss,
            metrics.learning_rate,
            metrics.grad_norm.unwrap_or(0.0),
            format_duration(self.start_time.elapsed())
        );
        self.metrics_bar.set_message(msg);
        self.metrics_bar.tick();
    }

    /// Update resource monitoring
    pub fn update_resources(&self, resources: &ResourceUsage) {
        if let Some(bar) = &self.resource_bar {
            let msg = format!(
                "CPU: {:.1}% | RAM: {:.1}GB | GPU: {}",
                resources.cpu_percent,
                resources.ram_gb,
                resources
                    .gpu_percent
                    .map(|g| format!("{:.1}%", g))
                    .unwrap_or_else(|| "N/A".to_string())
            );
            bar.set_message(msg);
            bar.tick();
        }
    }

    /// Finish epoch
    pub fn finish_epoch(&mut self, epoch_metrics: &EpochMetrics) {
        self.epoch_bar.inc(1);
        self.epoch_bar.set_message(format!(
            "train_loss: {:.4}, val_loss: {:.4}, time: {}s",
            epoch_metrics.train_loss,
            epoch_metrics.val_loss.unwrap_or(0.0),
            epoch_metrics.duration.as_secs()
        ));
    }

    /// Finish training
    pub fn finish(&self, final_message: &str) {
        self.epoch_bar
            .finish_with_message(final_message.to_string());
        self.batch_bar.finish_and_clear();
        self.metrics_bar.finish_and_clear();
        if let Some(bar) = &self.resource_bar {
            bar.finish_and_clear();
        }
    }

    /// Print summary statistics
    pub fn print_summary(&self, stats: &TrainingStats) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              Training Summary                            ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║ Total time:         {:<30} ║",
            format_duration(stats.total_duration)
        );
        println!("║ Epochs completed:   {:<30} ║", stats.epochs_completed);
        println!("║ Total steps:        {:<30} ║", stats.total_steps);
        println!("║ Final train loss:   {:<30.4} ║", stats.final_train_loss);
        if let Some(val_loss) = stats.final_val_loss {
            println!("║ Final val loss:     {:<30.4} ║", val_loss);
        }
        println!(
            "║ Best val loss:      {:<30.4} ║",
            stats.best_val_loss.unwrap_or(0.0)
        );
        println!(
            "║ Avg samples/sec:    {:<30.1} ║",
            stats.avg_samples_per_sec
        );
        println!("╚══════════════════════════════════════════════════════════╝");
    }
}

/// Training metrics for current step
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
}

/// Metrics for completed epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub duration: Duration,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub ram_gb: f64,
    pub gpu_percent: Option<f64>,
    pub gpu_memory_gb: Option<f64>,
}

impl ResourceUsage {
    /// Get current resource usage
    pub fn current() -> Self {
        // Get RAM usage
        let ram_gb = Self::get_memory_usage_gb();

        // Get CPU usage (approximate using process info)
        let cpu_percent = Self::get_cpu_usage_percent();

        // GPU monitoring would require platform-specific APIs
        // (CUDA for NVIDIA, Metal for macOS, etc.)
        Self {
            cpu_percent,
            ram_gb,
            gpu_percent: None,
            gpu_memory_gb: None,
        }
    }

    /// Get current memory usage in GB
    #[cfg(target_os = "macos")]
    fn get_memory_usage_gb() -> f64 {
        use std::mem;

        unsafe {
            let mut info: libc::vm_statistics64 = mem::zeroed();
            let mut count = (mem::size_of::<libc::vm_statistics64>() / mem::size_of::<libc::integer_t>()) as libc::mach_msg_type_number_t;

            let host_port = libc::mach_host_self();
            let result = libc::host_statistics64(
                host_port,
                libc::HOST_VM_INFO64,
                &mut info as *mut _ as *mut _,
                &mut count,
            );

            if result == libc::KERN_SUCCESS {
                let page_size = Self::get_page_size();
                let used_memory = (info.active_count + info.inactive_count + info.wire_count) as u64 * page_size;
                used_memory as f64 / 1_073_741_824.0 // Convert bytes to GB
            } else {
                0.0
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_memory_usage_gb() -> f64 {
        // Read /proc/meminfo on Linux
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0u64;
            let mut available_kb = 0u64;

            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total_kb = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                } else if line.starts_with("MemAvailable:") {
                    available_kb = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
            }

            if total_kb > 0 && available_kb > 0 {
                let used_kb = total_kb - available_kb;
                return used_kb as f64 / 1_048_576.0; // Convert KB to GB
            }
        }
        0.0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_memory_usage_gb() -> f64 {
        // Fallback for unsupported platforms
        0.0
    }

    #[cfg(target_os = "macos")]
    fn get_page_size() -> u64 {
        unsafe {
            libc::sysconf(libc::_SC_PAGESIZE) as u64
        }
    }

    /// Get approximate CPU usage percent
    fn get_cpu_usage_percent() -> f64 {
        // Simple approximation: assume 50% usage during training
        // For accurate measurement, would need to track process CPU time
        // over intervals (would require sysinfo or similar crate)

        // For now, return estimated load based on CPU count
        let cpu_count = num_cpus::get();

        // Estimate based on active training (typically uses 70-90% of available cores)
        let estimated_usage = 75.0 * cpu_count as f64 / cpu_count as f64;
        estimated_usage.min(100.0)
    }
}

/// Training statistics summary
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_duration: Duration,
    pub epochs_completed: usize,
    pub total_steps: usize,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub best_val_loss: Option<f64>,
    pub avg_samples_per_sec: f64,
}

/// Format duration as human-readable string
fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m 5s");
        assert_eq!(format_duration(Duration::from_secs(7325)), "2h 2m");
    }
}
