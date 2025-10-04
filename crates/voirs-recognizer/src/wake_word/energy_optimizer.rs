//! Energy optimization for wake word detection
//!
//! Provides energy-efficient detection algorithms for always-on listening
//! with battery optimization and adaptive processing.

use crate::RecognitionError;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Energy optimization configuration
#[derive(Debug, Clone)]
pub struct EnergyConfig {
    /// Enable energy saving mode
    pub energy_saving_enabled: bool,
    /// Minimum processing interval (ms)
    pub min_processing_interval_ms: u64,
    /// Maximum processing interval (ms)
    pub max_processing_interval_ms: u64,
    /// Audio level threshold for processing
    pub audio_level_threshold: f32,
    /// CPU usage threshold for throttling
    pub cpu_usage_threshold: f32,
    /// Battery level threshold for aggressive optimization
    pub battery_threshold: f32,
    /// Adaptive processing enabled
    pub adaptive_processing: bool,
    /// Background noise estimation enabled
    pub noise_estimation: bool,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            energy_saving_enabled: true,
            min_processing_interval_ms: 50,  // 50ms minimum
            max_processing_interval_ms: 500, // 500ms maximum
            audio_level_threshold: 0.01,     // Skip processing for very quiet audio
            cpu_usage_threshold: 0.8,        // Throttle if CPU > 80%
            battery_threshold: 0.2,          // Aggressive optimization if battery < 20%
            adaptive_processing: true,
            noise_estimation: true,
        }
    }
}

/// Energy optimization statistics
#[derive(Debug, Clone)]
pub struct EnergyStats {
    /// Total processing cycles
    pub total_cycles: u64,
    /// Skipped processing cycles
    pub skipped_cycles: u64,
    /// Average processing interval
    pub avg_processing_interval_ms: f32,
    /// Estimated power savings (percentage)
    pub power_savings_percent: f32,
    /// CPU usage statistics
    pub cpu_usage_stats: CpuUsageStats,
    /// Background noise level
    pub background_noise_level: f32,
}

/// CPU usage statistics
#[derive(Debug, Clone)]
pub struct CpuUsageStats {
    /// Current CPU usage (0.0 to 1.0)
    pub current_usage: f32,
    /// Average CPU usage over time
    pub avg_usage: f32,
    /// Peak CPU usage
    pub peak_usage: f32,
    /// CPU usage history (last 60 measurements)
    pub usage_history: Vec<f32>,
}

impl Default for CpuUsageStats {
    fn default() -> Self {
        Self {
            current_usage: 0.0,
            avg_usage: 0.0,
            peak_usage: 0.0,
            usage_history: Vec::new(),
        }
    }
}

impl Default for EnergyStats {
    fn default() -> Self {
        Self {
            total_cycles: 0,
            skipped_cycles: 0,
            avg_processing_interval_ms: 100.0,
            power_savings_percent: 0.0,
            cpu_usage_stats: CpuUsageStats::default(),
            background_noise_level: 0.0,
        }
    }
}

/// Energy optimizer for wake word detection
pub struct EnergyOptimizer {
    /// Configuration
    config: EnergyConfig,
    /// Statistics
    stats: Arc<Mutex<EnergyStats>>,
    /// Processing interval history
    interval_history: Arc<Mutex<VecDeque<Duration>>>,
    /// Last processing time
    last_processing: Arc<Mutex<Option<Instant>>>,
    /// Current processing interval
    current_interval: Arc<Mutex<Duration>>,
    /// Audio level history for noise estimation
    audio_level_history: Arc<Mutex<VecDeque<f32>>>,
    /// Detection history for adaptation
    detection_history: Arc<Mutex<VecDeque<DetectionEvent>>>,
    /// System resource monitor
    resource_monitor: Arc<Mutex<SystemResourceMonitor>>,
}

/// Detection event for learning
#[derive(Debug, Clone)]
struct DetectionEvent {
    /// Timestamp of detection
    timestamp: Instant,
    /// Whether detection was successful
    was_detection: bool,
    /// Audio level at time of detection
    audio_level: f32,
    /// Processing time taken
    processing_time: Duration,
}

/// System resource monitor
#[derive(Debug)]
struct SystemResourceMonitor {
    /// CPU usage measurements
    cpu_measurements: VecDeque<f32>,
    /// Memory usage measurements
    memory_measurements: VecDeque<f32>,
    /// Battery level (if available)
    battery_level: Option<f32>,
    /// Last measurement time
    last_measurement: Instant,
}

impl SystemResourceMonitor {
    fn new() -> Self {
        Self {
            cpu_measurements: VecDeque::new(),
            memory_measurements: VecDeque::new(),
            battery_level: None,
            last_measurement: Instant::now(),
        }
    }

    /// Update resource measurements
    fn update_measurements(&mut self) {
        let now = Instant::now();

        // Only update every second to avoid overhead
        if now.duration_since(self.last_measurement) < Duration::from_secs(1) {
            return;
        }

        // Simulate CPU usage measurement
        // In a real implementation, this would use system APIs
        let cpu_usage = Self::get_cpu_usage();
        self.cpu_measurements.push_back(cpu_usage);

        // Keep only last 60 measurements (1 minute of history)
        while self.cpu_measurements.len() > 60 {
            self.cpu_measurements.pop_front();
        }

        // Simulate memory usage
        let memory_usage = Self::get_memory_usage();
        self.memory_measurements.push_back(memory_usage);
        while self.memory_measurements.len() > 60 {
            self.memory_measurements.pop_front();
        }

        // Update battery level
        self.battery_level = Self::get_battery_level();

        self.last_measurement = now;
    }

    /// Get current CPU usage (simulated)
    fn get_cpu_usage() -> f32 {
        // Simulate CPU usage with some randomness
        thread_rng().r#gen::<f32>() * 0.3 + 0.1 // 10-40% usage
    }

    /// Get current memory usage (simulated)
    fn get_memory_usage() -> f32 {
        // Simulate memory usage
        thread_rng().r#gen::<f32>() * 0.2 + 0.4 // 40-60% usage
    }

    /// Get battery level (simulated)
    fn get_battery_level() -> Option<f32> {
        // Simulate battery level (would use platform-specific APIs)
        Some(thread_rng().r#gen::<f32>() * 0.6 + 0.4) // 40-100%
    }

    /// Get average CPU usage
    fn get_avg_cpu_usage(&self) -> f32 {
        if self.cpu_measurements.is_empty() {
            return 0.0;
        }
        self.cpu_measurements.iter().sum::<f32>() / self.cpu_measurements.len() as f32
    }

    /// Get current CPU usage
    fn get_current_cpu_usage(&self) -> f32 {
        self.cpu_measurements.back().copied().unwrap_or(0.0)
    }

    /// Get peak CPU usage
    fn get_peak_cpu_usage(&self) -> f32 {
        self.cpu_measurements.iter().fold(0.0, |acc, &x| acc.max(x))
    }
}

impl EnergyOptimizer {
    /// Create new energy optimizer
    pub fn new(energy_saving_enabled: bool) -> Self {
        let mut config = EnergyConfig::default();
        config.energy_saving_enabled = energy_saving_enabled;

        Self {
            config,
            stats: Arc::new(Mutex::new(EnergyStats::default())),
            interval_history: Arc::new(Mutex::new(VecDeque::new())),
            last_processing: Arc::new(Mutex::new(None)),
            current_interval: Arc::new(Mutex::new(Duration::from_millis(100))),
            audio_level_history: Arc::new(Mutex::new(VecDeque::new())),
            detection_history: Arc::new(Mutex::new(VecDeque::new())),
            resource_monitor: Arc::new(Mutex::new(SystemResourceMonitor::new())),
        }
    }

    /// Start energy optimization
    pub async fn start_optimization(&self) -> Result<(), RecognitionError> {
        if !self.config.energy_saving_enabled {
            return Ok(());
        }

        tracing::info!("Starting energy optimization for wake word detection");

        // Initialize resource monitoring
        {
            let mut monitor = self.resource_monitor.lock().unwrap();
            monitor.update_measurements();
        }

        Ok(())
    }

    /// Stop energy optimization
    pub async fn stop_optimization(&self) -> Result<(), RecognitionError> {
        tracing::info!("Stopping energy optimization");
        Ok(())
    }

    /// Check if processing should be skipped for energy saving
    pub async fn should_skip_processing(&self) -> bool {
        if !self.config.energy_saving_enabled {
            return false;
        }

        let now = Instant::now();

        // Check if enough time has passed since last processing
        {
            let last_processing = self.last_processing.lock().unwrap();
            let current_interval = *self.current_interval.lock().unwrap();

            if let Some(last) = *last_processing {
                if now.duration_since(last) < current_interval {
                    self.increment_skipped_cycles();
                    return true;
                }
            }
        }

        // Update resource measurements
        {
            let mut monitor = self.resource_monitor.lock().unwrap();
            monitor.update_measurements();
        }

        // Check CPU usage threshold
        if self.config.adaptive_processing {
            let cpu_usage = {
                let monitor = self.resource_monitor.lock().unwrap();
                monitor.get_current_cpu_usage()
            };

            if cpu_usage > self.config.cpu_usage_threshold {
                self.adapt_processing_interval(true).await;
                self.increment_skipped_cycles();
                return true;
            }
        }

        // Check battery level for aggressive optimization
        if let Some(battery_level) = self.get_battery_level() {
            if battery_level < self.config.battery_threshold {
                // More aggressive energy saving
                let extended_interval =
                    Duration::from_millis(self.config.max_processing_interval_ms * 2);
                {
                    let mut current_interval = self.current_interval.lock().unwrap();
                    *current_interval = extended_interval;
                }

                let last_processing = self.last_processing.lock().unwrap();
                if let Some(last) = *last_processing {
                    if now.duration_since(last) < extended_interval {
                        self.increment_skipped_cycles();
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Update processing result for adaptation
    pub async fn update_processing_result(&self, processing_time: Duration, had_detection: bool) {
        // Update last processing time
        {
            let mut last_processing = self.last_processing.lock().unwrap();
            *last_processing = Some(Instant::now());
        }

        // Record detection event
        let audio_level = self.get_current_audio_level();
        let event = DetectionEvent {
            timestamp: Instant::now(),
            was_detection: had_detection,
            audio_level,
            processing_time,
        };

        {
            let mut history = self.detection_history.lock().unwrap();
            history.push_back(event);

            // Keep only recent history (last 100 events)
            while history.len() > 100 {
                history.pop_front();
            }
        }

        // Update statistics
        self.update_stats(processing_time).await;

        // Adapt processing interval based on results
        if self.config.adaptive_processing {
            self.adapt_processing_interval(false).await;
        }
    }

    /// Adapt processing interval based on recent activity
    async fn adapt_processing_interval(&self, cpu_pressure: bool) {
        let mut should_increase_interval = cpu_pressure;
        let mut should_decrease_interval = false;

        // Analyze recent detection history
        {
            let history = self.detection_history.lock().unwrap();
            let recent_events: Vec<_> = history
                .iter()
                .filter(|event| event.timestamp.elapsed() < Duration::from_secs(30))
                .collect();

            if !recent_events.is_empty() {
                let detection_rate = recent_events
                    .iter()
                    .filter(|event| event.was_detection)
                    .count() as f32
                    / recent_events.len() as f32;

                // If we're detecting frequently, process more often
                if detection_rate > 0.1 && !cpu_pressure {
                    should_decrease_interval = true;
                }
                // If no detections, we can process less frequently
                else if detection_rate == 0.0 {
                    should_increase_interval = true;
                }
            }
        }

        // Adjust interval
        {
            let mut current_interval = self.current_interval.lock().unwrap();
            let current_ms = current_interval.as_millis() as u64;

            if should_decrease_interval {
                let new_ms = (current_ms * 3 / 4).max(self.config.min_processing_interval_ms);
                *current_interval = Duration::from_millis(new_ms);
            } else if should_increase_interval {
                let new_ms = (current_ms * 5 / 4).min(self.config.max_processing_interval_ms);
                *current_interval = Duration::from_millis(new_ms);
            }
        }
    }

    /// Update audio level for noise estimation
    pub async fn update_audio_level(&self, audio_level: f32) {
        if !self.config.noise_estimation {
            return;
        }

        let mut history = self.audio_level_history.lock().unwrap();
        history.push_back(audio_level);

        // Keep only recent history (last 300 measurements ~30 seconds at 10Hz)
        while history.len() > 300 {
            history.pop_front();
        }
    }

    /// Get current audio level estimate
    fn get_current_audio_level(&self) -> f32 {
        let history = self.audio_level_history.lock().unwrap();
        history.back().copied().unwrap_or(0.0)
    }

    /// Get background noise level estimate
    pub fn get_background_noise_level(&self) -> f32 {
        let history = self.audio_level_history.lock().unwrap();
        if history.is_empty() {
            return 0.0;
        }

        // Use 10th percentile as background noise estimate
        let mut sorted: Vec<f32> = history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_10_idx = (sorted.len() as f32 * 0.1) as usize;
        sorted.get(percentile_10_idx).copied().unwrap_or(0.0)
    }

    /// Get battery level
    fn get_battery_level(&self) -> Option<f32> {
        let monitor = self.resource_monitor.lock().unwrap();
        monitor.battery_level
    }

    /// Set energy saving mode
    pub async fn set_energy_saving(&mut self, enabled: bool) {
        self.config.energy_saving_enabled = enabled;

        if enabled {
            tracing::info!("Energy saving mode enabled");
        } else {
            tracing::info!("Energy saving mode disabled");
        }
    }

    /// Increment skipped cycles counter
    fn increment_skipped_cycles(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.skipped_cycles += 1;
        stats.total_cycles += 1;
    }

    /// Update energy statistics
    async fn update_stats(&self, _processing_time: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_cycles += 1;

        // Update processing interval
        {
            let mut interval_history = self.interval_history.lock().unwrap();
            let current_interval = *self.current_interval.lock().unwrap();
            interval_history.push_back(current_interval);

            // Keep only recent history
            while interval_history.len() > 60 {
                interval_history.pop_front();
            }

            // Calculate average
            if !interval_history.is_empty() {
                let total_ms: u64 = interval_history.iter().map(|d| d.as_millis() as u64).sum();
                stats.avg_processing_interval_ms = total_ms as f32 / interval_history.len() as f32;
            }
        }

        // Calculate power savings
        if stats.total_cycles > 0 {
            stats.power_savings_percent =
                (stats.skipped_cycles as f32 / stats.total_cycles as f32) * 100.0;
        }

        // Update CPU usage stats
        {
            let monitor = self.resource_monitor.lock().unwrap();
            stats.cpu_usage_stats.current_usage = monitor.get_current_cpu_usage();
            stats.cpu_usage_stats.avg_usage = monitor.get_avg_cpu_usage();
            stats.cpu_usage_stats.peak_usage = monitor.get_peak_cpu_usage();
            stats.cpu_usage_stats.usage_history =
                monitor.cpu_measurements.iter().copied().collect();
        }

        // Update background noise level
        stats.background_noise_level = self.get_background_noise_level();
    }

    /// Get energy optimization statistics
    pub async fn get_stats(&self) -> EnergyStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_energy_optimizer_creation() {
        let optimizer = EnergyOptimizer::new(true);
        assert!(optimizer.config.energy_saving_enabled);

        let stats = optimizer.get_stats().await;
        assert_eq!(stats.total_cycles, 0);
        assert_eq!(stats.skipped_cycles, 0);
    }

    #[tokio::test]
    async fn test_processing_skip_logic() {
        let optimizer = EnergyOptimizer::new(true);

        // First call should not skip (no previous processing)
        assert!(!optimizer.should_skip_processing().await);

        // Update with processing result
        optimizer
            .update_processing_result(Duration::from_millis(50), false)
            .await;

        // Immediate second call should skip due to interval
        assert!(optimizer.should_skip_processing().await);
    }

    #[tokio::test]
    async fn test_energy_saving_disabled() {
        let optimizer = EnergyOptimizer::new(false);

        // Should never skip when energy saving is disabled
        assert!(!optimizer.should_skip_processing().await);
        optimizer
            .update_processing_result(Duration::from_millis(50), false)
            .await;
        assert!(!optimizer.should_skip_processing().await);
    }

    #[tokio::test]
    async fn test_audio_level_tracking() {
        let optimizer = EnergyOptimizer::new(true);

        // Update audio levels
        optimizer.update_audio_level(0.1).await;
        optimizer.update_audio_level(0.05).await;
        optimizer.update_audio_level(0.2).await;

        let noise_level = optimizer.get_background_noise_level();
        assert!(noise_level >= 0.0 && noise_level <= 0.2);
    }

    #[tokio::test]
    async fn test_adaptive_processing() {
        let optimizer = EnergyOptimizer::new(true);

        // Simulate high CPU usage
        optimizer.adapt_processing_interval(true).await;

        let stats = optimizer.get_stats().await;
        assert!(stats.avg_processing_interval_ms >= 100.0);
    }

    #[test]
    fn test_system_resource_monitor() {
        let mut monitor = SystemResourceMonitor::new();
        monitor.update_measurements();

        let cpu_usage = monitor.get_current_cpu_usage();
        assert!(cpu_usage >= 0.0 && cpu_usage <= 1.0);

        let avg_usage = monitor.get_avg_cpu_usage();
        assert!(avg_usage >= 0.0);
    }
}
