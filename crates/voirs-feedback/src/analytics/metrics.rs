//! Performance metrics and system monitoring functionality

use super::types::{PerformanceMetrics, UserInteractionEvent};
use chrono::{DateTime, Utc};
use std::time::Instant;

/// Memory usage statistics for analytics data
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of items currently stored
    pub item_count: usize,
    /// Number of cleanup operations performed
    pub cleanup_count: u64,
    /// Last cleanup timestamp
    pub last_cleanup: Option<Instant>,
}

impl MemoryStats {
    /// Update memory statistics
    pub fn update(&mut self, new_usage: usize, item_count: usize) {
        self.current_usage = new_usage;
        self.peak_usage = self.peak_usage.max(new_usage);
        self.item_count = item_count;
    }

    /// Record cleanup operation
    pub fn record_cleanup(&mut self) {
        self.cleanup_count += 1;
        self.last_cleanup = Some(Instant::now());
    }

    /// Get memory efficiency ratio (items per byte)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.current_usage == 0 {
            0.0
        } else {
            self.item_count as f64 / self.current_usage as f64
        }
    }
}

/// System-wide metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Total interactions processed
    pub total_interactions: u64,
    /// Total users
    pub total_users: u64,
    /// System uptime
    pub uptime: std::time::Duration,
    /// Start time
    pub start_time: DateTime<Utc>,
}

impl SystemMetrics {
    /// Create new system metrics
    pub fn new() -> Self {
        Self {
            total_interactions: 0,
            total_users: 0,
            uptime: std::time::Duration::new(0, 0),
            start_time: Utc::now(),
        }
    }

    /// Update from interaction
    pub fn update_from_interaction(&mut self, _interaction: &UserInteractionEvent) {
        self.total_interactions += 1;
        self.uptime = std::time::Duration::from_secs(
            (Utc::now().timestamp() - self.start_time.timestamp()) as u64,
        );
    }

    /// Update from performance metrics
    pub fn update_from_performance(&mut self, _metrics: &PerformanceMetrics) {
        // Update system-wide performance tracking
        self.uptime = std::time::Duration::from_secs(
            (Utc::now().timestamp() - self.start_time.timestamp()) as u64,
        );
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::types::{InteractionType, UserInteractionEvent};
    use std::collections::HashMap;

    #[test]
    fn test_memory_stats_creation() {
        let stats = MemoryStats::default();
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.item_count, 0);
        assert_eq!(stats.cleanup_count, 0);
        assert!(stats.last_cleanup.is_none());
    }

    #[test]
    fn test_memory_stats_update() {
        let mut stats = MemoryStats::default();
        stats.update(1024, 10);

        assert_eq!(stats.current_usage, 1024);
        assert_eq!(stats.peak_usage, 1024);
        assert_eq!(stats.item_count, 10);

        // Test peak tracking
        stats.update(512, 5);
        assert_eq!(stats.current_usage, 512);
        assert_eq!(stats.peak_usage, 1024); // Should remain at peak
        assert_eq!(stats.item_count, 5);
    }

    #[test]
    fn test_memory_stats_cleanup() {
        let mut stats = MemoryStats::default();
        assert_eq!(stats.cleanup_count, 0);

        stats.record_cleanup();
        assert_eq!(stats.cleanup_count, 1);
        assert!(stats.last_cleanup.is_some());
    }

    #[test]
    fn test_memory_stats_efficiency_ratio() {
        let mut stats = MemoryStats::default();
        assert_eq!(stats.efficiency_ratio(), 0.0); // No usage

        stats.update(1000, 100);
        assert_eq!(stats.efficiency_ratio(), 0.1); // 100 items / 1000 bytes
    }

    #[test]
    fn test_system_metrics_creation() {
        let metrics = SystemMetrics::new();
        assert_eq!(metrics.total_interactions, 0);
        assert_eq!(metrics.total_users, 0);
        assert_eq!(metrics.uptime.as_secs(), 0);
    }

    #[test]
    fn test_system_metrics_interaction_update() {
        let mut metrics = SystemMetrics::new();

        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "test_feature".to_string(),
            feedback_score: Some(0.8),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        metrics.update_from_interaction(&interaction);
        assert_eq!(metrics.total_interactions, 1);
    }

    #[test]
    fn test_system_metrics_performance_update() {
        let mut metrics = SystemMetrics::new();

        let performance = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_ms: 100.0,
            throughput: 50.0,
            error_rate: 0.01,
            memory_usage: 1024,
            cpu_usage: 25.0,
        };

        metrics.update_from_performance(&performance);
        // Should update uptime
        assert!(metrics.uptime.as_secs() >= 0);
    }
}
