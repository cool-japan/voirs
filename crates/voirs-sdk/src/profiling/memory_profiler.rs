//! Memory profiling and allocation tracking.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

fn instant_now() -> Instant {
    Instant::now()
}

/// A snapshot of memory usage at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp when snapshot was taken
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,

    /// Total allocated memory in bytes
    pub allocated_bytes: usize,

    /// Total resident memory in bytes
    pub resident_bytes: usize,

    /// Peak memory usage in bytes
    pub peak_bytes: usize,

    /// Number of active allocations
    pub allocation_count: usize,

    /// Stage or operation associated with this snapshot
    pub context: Option<String>,
}

impl MemorySnapshot {
    /// Create a new memory snapshot.
    pub fn new(allocated: usize, resident: usize, peak: usize, count: usize) -> Self {
        Self {
            timestamp: Instant::now(),
            allocated_bytes: allocated,
            resident_bytes: resident,
            peak_bytes: peak,
            allocation_count: count,
            context: None,
        }
    }

    /// Create snapshot with context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Get memory usage in megabytes.
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get resident memory in megabytes.
    pub fn resident_mb(&self) -> f64 {
        self.resident_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory in megabytes.
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Tracks memory allocations and deallocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationTracker {
    /// Total bytes allocated
    pub total_allocated: usize,

    /// Total bytes deallocated
    pub total_deallocated: usize,

    /// Current active allocations
    pub active_allocations: usize,

    /// Peak allocation count
    pub peak_allocations: usize,

    /// Peak memory usage
    pub peak_memory_bytes: usize,

    /// Number of allocation events
    pub allocation_events: usize,

    /// Number of deallocation events
    pub deallocation_events: usize,
}

impl AllocationTracker {
    /// Create a new allocation tracker.
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            active_allocations: 0,
            peak_allocations: 0,
            peak_memory_bytes: 0,
            allocation_events: 0,
            deallocation_events: 0,
        }
    }

    /// Record an allocation.
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.active_allocations += 1;
        self.allocation_events += 1;

        if self.active_allocations > self.peak_allocations {
            self.peak_allocations = self.active_allocations;
        }

        let current_memory = self.total_allocated - self.total_deallocated;
        if current_memory > self.peak_memory_bytes {
            self.peak_memory_bytes = current_memory;
        }
    }

    /// Record a deallocation.
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_deallocated += size;
        if self.active_allocations > 0 {
            self.active_allocations -= 1;
        }
        self.deallocation_events += 1;
    }

    /// Get current memory usage.
    pub fn current_memory(&self) -> usize {
        self.total_allocated.saturating_sub(self.total_deallocated)
    }

    /// Get current memory usage in megabytes.
    pub fn current_memory_mb(&self) -> f64 {
        self.current_memory() as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in megabytes.
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get allocation efficiency (ratio of deallocations to allocations).
    pub fn allocation_efficiency(&self) -> f64 {
        if self.allocation_events == 0 {
            0.0
        } else {
            self.deallocation_events as f64 / self.allocation_events as f64
        }
    }

    /// Check if there are potential memory leaks.
    pub fn has_potential_leaks(&self) -> bool {
        // If we have significantly more allocations than deallocations
        let leak_threshold = 0.8; // 80% of allocations should be deallocated
        self.allocation_efficiency() < leak_threshold && self.allocation_events > 100
    }
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory profiler for tracking memory usage during synthesis.
pub struct MemoryProfiler {
    snapshots: Arc<RwLock<Vec<MemorySnapshot>>>,
    tracker: Arc<RwLock<AllocationTracker>>,
    baseline_snapshot: Arc<RwLock<Option<MemorySnapshot>>>,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(Vec::new())),
            tracker: Arc::new(RwLock::new(AllocationTracker::new())),
            baseline_snapshot: Arc::new(RwLock::new(None)),
        }
    }

    /// Take a memory snapshot.
    pub async fn take_snapshot(&self, context: Option<String>) -> MemorySnapshot {
        let tracker = self.tracker.read().await;

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            allocated_bytes: tracker.current_memory(),
            resident_bytes: tracker.current_memory(), // Simplified
            peak_bytes: tracker.peak_memory_bytes,
            allocation_count: tracker.active_allocations,
            context,
        };

        let mut snapshots = self.snapshots.write().await;
        snapshots.push(snapshot.clone());

        snapshot
    }

    /// Set baseline snapshot for comparison.
    pub async fn set_baseline(&self, snapshot: MemorySnapshot) {
        let mut baseline = self.baseline_snapshot.write().await;
        *baseline = Some(snapshot);
    }

    /// Get memory growth since baseline.
    pub async fn get_memory_growth(&self) -> Option<i64> {
        let baseline = self.baseline_snapshot.read().await;
        let tracker = self.tracker.read().await;

        if let Some(baseline_snapshot) = baseline.as_ref() {
            let current = tracker.current_memory() as i64;
            let baseline = baseline_snapshot.allocated_bytes as i64;
            Some(current - baseline)
        } else {
            None
        }
    }

    /// Get all memory snapshots.
    pub async fn get_snapshots(&self) -> Vec<MemorySnapshot> {
        self.snapshots.read().await.clone()
    }

    /// Get allocation tracker.
    pub async fn get_tracker(&self) -> AllocationTracker {
        self.tracker.read().await.clone()
    }

    /// Record allocation (for testing purposes).
    pub async fn record_allocation(&self, size: usize) {
        let mut tracker = self.tracker.write().await;
        tracker.record_allocation(size);
    }

    /// Record deallocation (for testing purposes).
    pub async fn record_deallocation(&self, size: usize) {
        let mut tracker = self.tracker.write().await;
        tracker.record_deallocation(size);
    }

    /// Get peak memory usage across all snapshots.
    pub async fn get_peak_memory(&self) -> usize {
        let snapshots = self.snapshots.read().await;
        snapshots
            .iter()
            .map(|s| s.allocated_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Get average memory usage.
    pub async fn get_average_memory(&self) -> f64 {
        let snapshots = self.snapshots.read().await;
        if snapshots.is_empty() {
            return 0.0;
        }

        let total: usize = snapshots.iter().map(|s| s.allocated_bytes).sum();
        total as f64 / snapshots.len() as f64
    }

    /// Reset profiler state.
    pub async fn reset(&self) {
        let mut snapshots = self.snapshots.write().await;
        snapshots.clear();
        let mut tracker = self.tracker.write().await;
        *tracker = AllocationTracker::new();
        let mut baseline = self.baseline_snapshot.write().await;
        *baseline = None;
    }

    /// Get memory usage summary.
    pub async fn get_summary(&self) -> MemorySummary {
        let tracker = self.tracker.read().await;
        let snapshots = self.snapshots.read().await;

        let peak_memory = snapshots
            .iter()
            .map(|s| s.allocated_bytes)
            .max()
            .unwrap_or(0);

        let avg_memory = if !snapshots.is_empty() {
            let total: usize = snapshots.iter().map(|s| s.allocated_bytes).sum();
            total as f64 / snapshots.len() as f64
        } else {
            0.0
        };

        MemorySummary {
            current_memory: tracker.current_memory(),
            peak_memory,
            average_memory: avg_memory,
            total_allocated: tracker.total_allocated,
            total_deallocated: tracker.total_deallocated,
            active_allocations: tracker.active_allocations,
            has_potential_leaks: tracker.has_potential_leaks(),
            allocation_efficiency: tracker.allocation_efficiency(),
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of memory profiling results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// Current memory usage in bytes
    pub current_memory: usize,

    /// Peak memory usage in bytes
    pub peak_memory: usize,

    /// Average memory usage in bytes
    pub average_memory: f64,

    /// Total allocated bytes
    pub total_allocated: usize,

    /// Total deallocated bytes
    pub total_deallocated: usize,

    /// Active allocation count
    pub active_allocations: usize,

    /// Whether potential memory leaks detected
    pub has_potential_leaks: bool,

    /// Allocation efficiency (0.0-1.0)
    pub allocation_efficiency: f64,
}

impl MemorySummary {
    /// Get current memory in megabytes.
    pub fn current_mb(&self) -> f64 {
        self.current_memory as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory in megabytes.
    pub fn peak_mb(&self) -> f64 {
        self.peak_memory as f64 / (1024.0 * 1024.0)
    }

    /// Get average memory in megabytes.
    pub fn average_mb(&self) -> f64 {
        self.average_memory / (1024.0 * 1024.0)
    }
}

impl fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory Profile Summary:\n\
             - Current: {:.2} MB\n\
             - Peak: {:.2} MB\n\
             - Average: {:.2} MB\n\
             - Total Allocated: {:.2} MB\n\
             - Total Deallocated: {:.2} MB\n\
             - Active Allocations: {}\n\
             - Allocation Efficiency: {:.1}%\n\
             - Potential Leaks: {}",
            self.current_mb(),
            self.peak_mb(),
            self.average_mb(),
            self.total_allocated as f64 / (1024.0 * 1024.0),
            self.total_deallocated as f64 / (1024.0 * 1024.0),
            self.active_allocations,
            self.allocation_efficiency * 100.0,
            if self.has_potential_leaks {
                "Yes"
            } else {
                "No"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot_creation() {
        let snapshot = MemorySnapshot::new(1024, 2048, 4096, 10);
        assert_eq!(snapshot.allocated_bytes, 1024);
        assert_eq!(snapshot.resident_bytes, 2048);
        assert_eq!(snapshot.peak_bytes, 4096);
        assert_eq!(snapshot.allocation_count, 10);
    }

    #[test]
    fn test_memory_snapshot_mb_conversion() {
        let snapshot = MemorySnapshot::new(1024 * 1024, 0, 0, 0);
        assert_eq!(snapshot.allocated_mb(), 1.0);
    }

    #[test]
    fn test_allocation_tracker() {
        let mut tracker = AllocationTracker::new();
        tracker.record_allocation(1024);
        tracker.record_allocation(2048);

        assert_eq!(tracker.total_allocated, 3072);
        assert_eq!(tracker.active_allocations, 2);
        assert_eq!(tracker.allocation_events, 2);

        tracker.record_deallocation(1024);
        assert_eq!(tracker.active_allocations, 1);
        assert_eq!(tracker.current_memory(), 2048);
    }

    #[test]
    fn test_allocation_efficiency() {
        let mut tracker = AllocationTracker::new();
        for _ in 0..10 {
            tracker.record_allocation(1024);
        }
        for _ in 0..8 {
            tracker.record_deallocation(1024);
        }

        assert_eq!(tracker.allocation_efficiency(), 0.8);
    }

    #[tokio::test]
    async fn test_memory_profiler_snapshot() {
        let profiler = MemoryProfiler::new();
        let snapshot = profiler.take_snapshot(Some("test".to_string())).await;

        assert_eq!(snapshot.context, Some("test".to_string()));

        let snapshots = profiler.get_snapshots().await;
        assert_eq!(snapshots.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_profiler_tracking() {
        let profiler = MemoryProfiler::new();

        profiler.record_allocation(1024).await;
        profiler.record_allocation(2048).await;

        let tracker = profiler.get_tracker().await;
        assert_eq!(tracker.total_allocated, 3072);
        assert_eq!(tracker.active_allocations, 2);
    }

    #[tokio::test]
    async fn test_memory_profiler_reset() {
        let profiler = MemoryProfiler::new();

        profiler.record_allocation(1024).await;
        profiler.take_snapshot(None).await;

        profiler.reset().await;

        let snapshots = profiler.get_snapshots().await;
        assert_eq!(snapshots.len(), 0);

        let tracker = profiler.get_tracker().await;
        assert_eq!(tracker.total_allocated, 0);
    }
}
