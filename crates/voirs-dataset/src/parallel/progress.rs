//! Progress tracking and monitoring for parallel dataset operations
//!
//! Provides real-time progress reporting, ETA calculation, throughput monitoring,
//! and error aggregation for long-running parallel tasks.

use crate::{DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio::time;
use tracing::{info, warn};

/// Progress update event
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Task ID
    pub task_id: String,
    /// Total items to process
    pub total_items: usize,
    /// Items completed
    pub completed_items: usize,
    /// Items failed
    pub failed_items: usize,
    /// Current progress (0.0 - 1.0)
    pub progress: f64,
    /// Estimated time remaining
    pub eta: Option<Duration>,
    /// Current throughput (items/second)
    pub throughput: f64,
    /// Average execution time per item
    pub avg_time_per_item: Duration,
    /// Status message
    pub status: String,
    /// Timestamp
    pub timestamp: Instant,
}

/// Progress tracking stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgressStage {
    /// Initialization phase
    Initializing,
    /// Loading data
    Loading,
    /// Processing data
    Processing,
    /// Validating results
    Validating,
    /// Finalizing and cleanup
    Finalizing,
    /// Completed successfully
    Completed,
    /// Failed with errors
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl ProgressStage {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Initializing => "Initializing",
            Self::Loading => "Loading data",
            Self::Processing => "Processing",
            Self::Validating => "Validating",
            Self::Finalizing => "Finalizing",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    /// Check if stage is terminal (finished)
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Error information for tracking
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Error message
    pub message: String,
    /// Error type/category
    pub error_type: String,
    /// Item index where error occurred
    pub item_index: Option<usize>,
    /// Timestamp when error occurred
    pub timestamp: Instant,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Throughput measurement window
#[derive(Debug, Clone)]
struct ThroughputWindow {
    /// Measurements: (timestamp, items_completed)
    measurements: Vec<(Instant, usize)>,
    /// Window size for measurements
    window_size: Duration,
}

impl ThroughputWindow {
    fn new(window_size: Duration) -> Self {
        Self {
            measurements: Vec::new(),
            window_size,
        }
    }

    fn add_measurement(&mut self, timestamp: Instant, items_completed: usize) {
        self.measurements.push((timestamp, items_completed));

        // Remove old measurements outside the window
        let cutoff = timestamp - self.window_size;
        self.measurements.retain(|(ts, _)| *ts >= cutoff);
    }

    fn calculate_throughput(&self) -> f64 {
        if self.measurements.len() < 2 {
            return 0.0;
        }

        let first = &self.measurements[0];
        let last = &self.measurements[self.measurements.len() - 1];

        let time_diff = last.0.duration_since(first.0).as_secs_f64();
        let items_diff = last.1.saturating_sub(first.1);

        if time_diff > 0.0 {
            items_diff as f64 / time_diff
        } else {
            0.0
        }
    }
}

/// Progress tracker for monitoring task execution
pub struct ProgressTracker {
    /// Task identifier
    task_id: String,
    /// Total number of items to process
    total_items: AtomicUsize,
    /// Number of completed items
    completed_items: AtomicUsize,
    /// Number of failed items
    failed_items: AtomicUsize,
    /// Current stage
    current_stage: RwLock<ProgressStage>,
    /// Start time
    start_time: Instant,
    /// Last update time
    last_update: RwLock<Instant>,
    /// Errors collected during processing
    errors: RwLock<Vec<ErrorInfo>>,
    /// Custom metadata
    metadata: RwLock<HashMap<String, String>>,
    /// Throughput tracking
    throughput_window: RwLock<ThroughputWindow>,
    /// Broadcast channel for progress updates
    update_sender: broadcast::Sender<ProgressUpdate>,
    /// Progress update receiver for internal use
    _update_receiver: broadcast::Receiver<ProgressUpdate>,
    /// Configuration
    config: ProgressConfig,
}

/// Progress tracking configuration
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    /// Minimum interval between progress updates
    pub update_interval: Duration,
    /// Throughput calculation window size
    pub throughput_window: Duration,
    /// Maximum number of errors to track
    pub max_errors: usize,
    /// Enable detailed error tracking
    pub detailed_errors: bool,
    /// Auto-send updates on changes
    pub auto_updates: bool,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(500), // 2 updates per second
            throughput_window: Duration::from_secs(30),  // 30-second window
            max_errors: 1000,
            detailed_errors: true,
            auto_updates: true,
        }
    }
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(task_id: String, total_items: usize) -> Self {
        Self::with_config(task_id, total_items, ProgressConfig::default())
    }

    /// Create a new progress tracker with custom configuration
    pub fn with_config(task_id: String, total_items: usize, config: ProgressConfig) -> Self {
        let (tx, rx) = broadcast::channel(1000);

        Self {
            task_id,
            total_items: AtomicUsize::new(total_items),
            completed_items: AtomicUsize::new(0),
            failed_items: AtomicUsize::new(0),
            current_stage: RwLock::new(ProgressStage::Initializing),
            start_time: Instant::now(),
            last_update: RwLock::new(Instant::now()),
            errors: RwLock::new(Vec::new()),
            metadata: RwLock::new(HashMap::new()),
            throughput_window: RwLock::new(ThroughputWindow::new(config.throughput_window)),
            update_sender: tx,
            _update_receiver: rx,
            config,
        }
    }

    /// Set the current stage
    pub fn set_stage(&self, stage: ProgressStage) {
        if let Ok(mut current_stage) = self.current_stage.write() {
            if *current_stage != stage {
                *current_stage = stage;
                info!("Task {}: Stage changed to {:?}", self.task_id, stage);

                if self.config.auto_updates {
                    let _ = self.send_update();
                }
            }
        }
    }

    /// Get the current stage
    pub fn get_stage(&self) -> ProgressStage {
        self.current_stage
            .read()
            .map(|stage| *stage)
            .unwrap_or(ProgressStage::Failed)
    }

    /// Update total items count
    pub fn set_total_items(&self, total: usize) {
        self.total_items.store(total, Ordering::Relaxed);

        if self.config.auto_updates {
            let _ = self.send_update();
        }
    }

    /// Increment completed items
    pub fn increment_completed(&self) -> usize {
        let completed = self.completed_items.fetch_add(1, Ordering::Relaxed) + 1;

        // Update throughput window
        if let Ok(mut window) = self.throughput_window.write() {
            window.add_measurement(Instant::now(), completed);
        }

        if self.config.auto_updates {
            let _ = self.send_update_if_needed();
        }

        completed
    }

    /// Increment completed items by a specific amount
    pub fn add_completed(&self, count: usize) -> usize {
        let completed = self.completed_items.fetch_add(count, Ordering::Relaxed) + count;

        // Update throughput window
        if let Ok(mut window) = self.throughput_window.write() {
            window.add_measurement(Instant::now(), completed);
        }

        if self.config.auto_updates {
            let _ = self.send_update_if_needed();
        }

        completed
    }

    /// Increment failed items
    pub fn increment_failed(&self) -> usize {
        let failed = self.failed_items.fetch_add(1, Ordering::Relaxed) + 1;

        if self.config.auto_updates {
            let _ = self.send_update_if_needed();
        }

        failed
    }

    /// Record an error
    pub fn record_error(
        &self,
        error: DatasetError,
        item_index: Option<usize>,
        context: HashMap<String, String>,
    ) {
        let error_info = ErrorInfo {
            message: error.to_string(),
            error_type: format!("{error:?}")
                .split('(')
                .next()
                .unwrap_or("Unknown")
                .to_string(),
            item_index,
            timestamp: Instant::now(),
            context,
        };

        if let Ok(mut errors) = self.errors.write() {
            if errors.len() < self.config.max_errors {
                errors.push(error_info);
            } else if self.config.max_errors > 0 {
                // Replace oldest error
                errors.remove(0);
                errors.push(error_info);
            }
        }

        self.increment_failed();

        warn!(
            "Task {}: Error recorded at item {:?}: {}",
            self.task_id, item_index, error
        );
    }

    /// Set custom metadata
    pub fn set_metadata(&self, key: String, value: String) {
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.insert(key, value);
        }
    }

    /// Get current progress (0.0 - 1.0)
    pub fn get_progress(&self) -> f64 {
        let total = self.total_items.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }

        let completed = self.completed_items.load(Ordering::Relaxed);
        completed as f64 / total as f64
    }

    /// Calculate estimated time remaining
    pub fn calculate_eta(&self) -> Option<Duration> {
        let progress = self.get_progress();
        if progress <= 0.0 || progress >= 1.0 {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let total_estimated = elapsed.as_secs_f64() / progress;
        let remaining = total_estimated - elapsed.as_secs_f64();

        if remaining > 0.0 {
            Some(Duration::from_secs_f64(remaining))
        } else {
            None
        }
    }

    /// Get current throughput (items/second)
    pub fn get_throughput(&self) -> f64 {
        if let Ok(window) = self.throughput_window.read() {
            window.calculate_throughput()
        } else {
            0.0
        }
    }

    /// Get average time per item
    pub fn get_avg_time_per_item(&self) -> Duration {
        let completed = self.completed_items.load(Ordering::Relaxed);
        if completed == 0 {
            return Duration::ZERO;
        }

        let elapsed = self.start_time.elapsed();
        Duration::from_nanos((elapsed.as_nanos() / completed as u128) as u64)
    }

    /// Get error summary
    pub fn get_error_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();

        if let Ok(errors) = self.errors.read() {
            for error in errors.iter() {
                *summary.entry(error.error_type.clone()).or_insert(0) += 1;
            }
        }

        summary
    }

    /// Get all recorded errors
    pub fn get_errors(&self) -> Vec<ErrorInfo> {
        self.errors
            .read()
            .map(|errors| errors.clone())
            .unwrap_or_else(|_| Vec::new())
    }

    /// Create a progress update
    pub fn create_update(&self) -> ProgressUpdate {
        let total = self.total_items.load(Ordering::Relaxed);
        let completed = self.completed_items.load(Ordering::Relaxed);
        let failed = self.failed_items.load(Ordering::Relaxed);
        let stage = self.get_stage();

        ProgressUpdate {
            task_id: self.task_id.clone(),
            total_items: total,
            completed_items: completed,
            failed_items: failed,
            progress: self.get_progress(),
            eta: self.calculate_eta(),
            throughput: self.get_throughput(),
            avg_time_per_item: self.get_avg_time_per_item(),
            status: format!("{} ({}/{})", stage.description(), completed, total),
            timestamp: Instant::now(),
        }
    }

    /// Send a progress update
    pub fn send_update(&self) -> Result<()> {
        let update = self.create_update();

        // Update last update time
        if let Ok(mut last_update) = self.last_update.write() {
            *last_update = Instant::now();
        }

        // Send update (ignore if no receivers)
        let _ = self.update_sender.send(update);

        Ok(())
    }

    /// Send update only if enough time has passed
    fn send_update_if_needed(&self) -> Result<()> {
        let should_update = if let Ok(last_update) = self.last_update.read() {
            last_update.elapsed() >= self.config.update_interval
        } else {
            true
        };

        if should_update {
            self.send_update()
        } else {
            Ok(())
        }
    }

    /// Subscribe to progress updates
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressUpdate> {
        self.update_sender.subscribe()
    }

    /// Mark task as completed
    pub fn complete(&self) {
        self.set_stage(ProgressStage::Completed);
        let _ = self.send_update();

        let elapsed = self.start_time.elapsed();
        let completed = self.completed_items.load(Ordering::Relaxed);
        let failed = self.failed_items.load(Ordering::Relaxed);

        info!(
            "Task {} completed: {} items processed, {} failed in {:?}",
            self.task_id, completed, failed, elapsed
        );
    }

    /// Mark task as failed
    pub fn fail(&self, reason: &str) {
        self.set_stage(ProgressStage::Failed);
        self.set_metadata("failure_reason".to_string(), reason.to_string());
        let _ = self.send_update();

        warn!("Task {} failed: {}", self.task_id, reason);
    }

    /// Mark task as cancelled
    pub fn cancel(&self) {
        self.set_stage(ProgressStage::Cancelled);
        let _ = self.send_update();

        info!("Task {} cancelled", self.task_id);
    }

    /// Check if task is finished
    pub fn is_finished(&self) -> bool {
        self.get_stage().is_terminal()
    }

    /// Get task duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get task ID
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Get configuration
    pub fn config(&self) -> &ProgressConfig {
        &self.config
    }
}

/// Progress manager for multiple concurrent tasks
pub struct ProgressManager {
    /// Active progress trackers
    trackers: RwLock<HashMap<String, Arc<ProgressTracker>>>,
    /// Global update broadcaster
    update_sender: broadcast::Sender<ProgressUpdate>,
    /// Configuration
    config: ProgressConfig,
}

impl ProgressManager {
    /// Create a new progress manager
    pub fn new() -> Self {
        Self::with_config(ProgressConfig::default())
    }

    /// Create a new progress manager with custom configuration
    pub fn with_config(config: ProgressConfig) -> Self {
        let (tx, _rx) = broadcast::channel(10000);

        Self {
            trackers: RwLock::new(HashMap::new()),
            update_sender: tx,
            config,
        }
    }

    /// Create a new progress tracker
    pub fn create_tracker(&self, task_id: String, total_items: usize) -> Arc<ProgressTracker> {
        let tracker = Arc::new(ProgressTracker::with_config(
            task_id.clone(),
            total_items,
            self.config.clone(),
        ));

        if let Ok(mut trackers) = self.trackers.write() {
            trackers.insert(task_id, Arc::clone(&tracker));
        }

        // Forward updates to global broadcaster with timeout
        let global_sender = self.update_sender.clone();
        let mut updates = tracker.subscribe();
        let tracker_weak = Arc::downgrade(&tracker);

        tokio::spawn(async move {
            loop {
                // Check if tracker is still alive
                if tracker_weak.upgrade().is_none() {
                    break;
                }

                // Use timeout to prevent hanging
                match time::timeout(Duration::from_secs(5), updates.recv()).await {
                    Ok(Ok(update)) => {
                        // Check for terminal state before sending
                        let is_terminal = matches!(update.status.as_str(), s if s.contains("Completed") || s.contains("Failed") || s.contains("Cancelled"));
                        let _ = global_sender.send(update);
                        // If update indicates terminal state, break
                        if is_terminal {
                            break;
                        }
                    }
                    Ok(Err(_)) => break, // Channel closed
                    Err(_) => {
                        // Timeout - check if tracker is finished
                        if let Some(tracker) = tracker_weak.upgrade() {
                            if tracker.is_finished() {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        });

        tracker
    }

    /// Get a progress tracker by ID
    pub fn get_tracker(&self, task_id: &str) -> Option<Arc<ProgressTracker>> {
        self.trackers.read().ok()?.get(task_id).cloned()
    }

    /// Remove a finished tracker
    pub fn remove_tracker(&self, task_id: &str) -> Option<Arc<ProgressTracker>> {
        self.trackers.write().ok()?.remove(task_id)
    }

    /// Get all active tracker IDs
    pub fn get_active_tasks(&self) -> Vec<String> {
        self.trackers
            .read()
            .map(|trackers| trackers.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Subscribe to all progress updates
    pub fn subscribe_all(&self) -> broadcast::Receiver<ProgressUpdate> {
        self.update_sender.subscribe()
    }

    /// Clean up finished trackers
    pub fn cleanup_finished(&self) {
        if let Ok(mut trackers) = self.trackers.write() {
            trackers.retain(|_id, tracker| !tracker.is_finished());
        }
    }

    /// Get summary of all active tasks
    pub fn get_summary(&self) -> HashMap<String, ProgressUpdate> {
        let mut summary = HashMap::new();

        if let Ok(trackers) = self.trackers.read() {
            for (id, tracker) in trackers.iter() {
                summary.insert(id.clone(), tracker.create_update());
            }
        }

        summary
    }

    /// Get configuration
    pub fn config(&self) -> &ProgressConfig {
        &self.config
    }
}

impl Default for ProgressManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_progress_tracker_basic() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let tracker = ProgressTracker::with_config("test-task".to_string(), 100, config);

        assert_eq!(tracker.task_id(), "test-task");
        assert_eq!(tracker.get_progress(), 0.0);
        assert_eq!(tracker.get_stage(), ProgressStage::Initializing);

        tracker.set_stage(ProgressStage::Processing);
        assert_eq!(tracker.get_stage(), ProgressStage::Processing);

        tracker.increment_completed();
        tracker.increment_completed();
        assert_eq!(tracker.get_progress(), 0.02); // 2/100

        tracker.complete();
        assert_eq!(tracker.get_stage(), ProgressStage::Completed);
        assert!(tracker.is_finished());
    }

    #[tokio::test]
    async fn test_progress_updates() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let tracker = ProgressTracker::with_config("test-updates".to_string(), 10, config);
        let mut receiver = tracker.subscribe();

        // Start processing
        tracker.set_stage(ProgressStage::Processing);

        // Trigger some progress
        tracker.increment_completed();
        tracker.increment_completed();

        // Send explicit update
        tracker.send_update().unwrap();

        // Receive update with timeout
        let update = time::timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(update.task_id, "test-updates");
        assert_eq!(update.completed_items, 2);
        assert_eq!(update.total_items, 10);
        assert_eq!(update.progress, 0.2);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let tracker = ProgressTracker::new("test-errors".to_string(), 5);

        let error = DatasetError::LoadError("Test error".to_string());
        let mut context = HashMap::new();
        context.insert("file".to_string(), "test.wav".to_string());

        tracker.record_error(error, Some(0), context);

        let errors = tracker.get_errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "Dataset loading failed: Test error");
        assert_eq!(errors[0].item_index, Some(0));

        let summary = tracker.get_error_summary();
        // The error type should be "LoadError" since that's the specific variant
        assert_eq!(summary.get("LoadError"), Some(&1));
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let tracker = ProgressTracker::new("test-throughput".to_string(), 100);

        // Process some items with delays to simulate real work
        for _ in 0..5 {
            tracker.increment_completed();
            sleep(Duration::from_millis(10)).await;
        }

        let throughput = tracker.get_throughput();
        assert!(throughput > 0.0);

        let avg_time = tracker.get_avg_time_per_item();
        assert!(avg_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_eta_calculation() {
        let tracker = ProgressTracker::new("test-eta".to_string(), 100);

        // Complete 25% of the work
        for _ in 0..25 {
            tracker.increment_completed();
        }

        // Give some time for calculation
        sleep(Duration::from_millis(100)).await;

        let eta = tracker.calculate_eta();
        assert!(eta.is_some());
        assert!(eta.unwrap() > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_progress_manager() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let manager = ProgressManager::with_config(config);

        let tracker1 = manager.create_tracker("task-1".to_string(), 50);
        let _tracker2 = manager.create_tracker("task-2".to_string(), 30);

        assert_eq!(manager.get_active_tasks().len(), 2);

        let retrieved = manager.get_tracker("task-1").unwrap();
        assert_eq!(retrieved.task_id(), "task-1");

        // Complete all items first, then mark as completed
        for _ in 0..50 {
            tracker1.increment_completed();
        }
        tracker1.complete();

        // Small delay to allow for async cleanup
        time::sleep(Duration::from_millis(100)).await;

        let summary = manager.get_summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary["task-1"].progress, 1.0);
    }

    #[tokio::test]
    async fn test_stage_transitions() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let tracker = ProgressTracker::with_config("test-stages".to_string(), 10, config);

        let stages = [
            ProgressStage::Initializing,
            ProgressStage::Loading,
            ProgressStage::Processing,
            ProgressStage::Validating,
            ProgressStage::Finalizing,
            ProgressStage::Completed,
        ];

        for stage in stages {
            tracker.set_stage(stage);
            assert_eq!(tracker.get_stage(), stage);

            if stage.is_terminal() {
                assert!(tracker.is_finished());
                break;
            } else {
                assert!(!tracker.is_finished());
            }
        }
    }

    #[tokio::test]
    async fn test_batch_progress_updates() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let tracker = ProgressTracker::with_config("test-batch".to_string(), 1000, config);

        // Process items in batches
        tracker.add_completed(100);
        tracker.add_completed(200);
        tracker.add_completed(300);

        assert_eq!(tracker.get_progress(), 0.6); // 600/1000

        let update = tracker.create_update();
        assert_eq!(update.completed_items, 600);
        assert_eq!(update.progress, 0.6);
    }

    #[tokio::test]
    async fn test_metadata_tracking() {
        let config = ProgressConfig {
            auto_updates: false, // Disable auto updates to prevent hanging
            ..Default::default()
        };
        let tracker = ProgressTracker::with_config("test-metadata".to_string(), 10, config);

        tracker.set_metadata("dataset".to_string(), "test-dataset".to_string());
        tracker.set_metadata("format".to_string(), "wav".to_string());

        // Metadata is internal, but we can test it doesn't crash
        tracker.fail("Test failure");

        assert_eq!(tracker.get_stage(), ProgressStage::Failed);
    }
}
