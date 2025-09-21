//! UI-Responsive Queue Management System
//!
//! This module provides non-blocking queue operations to ensure UI responsiveness
//! while handling real-time feedback processing.

use crate::FeedbackError;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::time::timeout;

/// UI-responsive queue for handling feedback operations without blocking
#[derive(Debug)]
pub struct UiResponsiveQueue<T> {
    /// Internal queue storage
    queue: Arc<RwLock<VecDeque<T>>>,
    /// Notification system for new items
    notify: Arc<Notify>,
    /// Maximum queue size to prevent memory issues
    max_size: usize,
    /// Processing timeout for UI operations
    ui_timeout: Duration,
    /// Queue statistics
    stats: Arc<RwLock<QueueStats>>,
}

/// Queue performance statistics
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total items processed
    pub items_processed: u64,
    /// Items currently in queue
    pub current_size: usize,
    /// Maximum queue size reached
    pub max_size_reached: usize,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Number of timeouts occurred
    pub timeouts: u64,
    /// Last update timestamp
    pub last_update: Option<Instant>,
}

/// Queue operation result
#[derive(Debug)]
pub enum QueueResult<T> {
    /// Operation completed successfully
    Success(T),
    /// Operation timed out (UI responsiveness preserved)
    Timeout,
    /// Queue is at capacity
    AtCapacity,
    /// Queue is empty
    Empty,
}

impl<T> UiResponsiveQueue<T>
where
    T: Send + Sync + 'static,
{
    /// Create a new UI-responsive queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Arc::new(RwLock::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            max_size,
            ui_timeout: Duration::from_millis(50), // 50ms max delay for UI
            stats: Arc::new(RwLock::new(QueueStats::default())),
        }
    }

    /// Create a queue with custom UI timeout
    pub fn with_ui_timeout(max_size: usize, ui_timeout: Duration) -> Self {
        Self {
            queue: Arc::new(RwLock::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            max_size,
            ui_timeout,
            stats: Arc::new(RwLock::new(QueueStats::default())),
        }
    }

    /// Add item to queue without blocking UI
    pub async fn enqueue_non_blocking(&self, item: T) -> QueueResult<()> {
        let start = Instant::now();

        // Try to acquire write lock with timeout
        let queue_lock = match timeout(self.ui_timeout, self.queue.write()).await {
            Ok(lock) => lock,
            Err(_) => return QueueResult::Timeout,
        };

        // Check capacity
        if queue_lock.len() >= self.max_size {
            return QueueResult::AtCapacity;
        }

        // Add item and update stats
        let mut queue = queue_lock;
        queue.push_back(item);
        let current_size = queue.len();
        drop(queue);

        // Update statistics
        if let Ok(mut stats) = self.stats.try_write() {
            stats.current_size = current_size;
            stats.max_size_reached = stats.max_size_reached.max(current_size);
            stats.last_update = Some(start);
        }

        // Notify waiting consumers
        self.notify.notify_one();
        QueueResult::Success(())
    }

    /// Remove item from queue without blocking UI
    pub async fn dequeue_non_blocking(&self) -> QueueResult<T> {
        let start = Instant::now();

        // Try to acquire write lock with timeout
        let mut queue_lock = match timeout(self.ui_timeout, self.queue.write()).await {
            Ok(lock) => lock,
            Err(_) => return QueueResult::Timeout,
        };

        // Try to get item
        match queue_lock.pop_front() {
            Some(item) => {
                let current_size = queue_lock.len();
                drop(queue_lock);

                // Update statistics
                if let Ok(mut stats) = self.stats.try_write() {
                    stats.current_size = current_size;
                    stats.items_processed += 1;
                    let processing_time = start.elapsed();

                    // Update average processing time
                    if stats.items_processed == 1 {
                        stats.avg_processing_time = processing_time;
                    } else {
                        let total_time = stats.avg_processing_time
                            * (stats.items_processed - 1) as u32
                            + processing_time;
                        stats.avg_processing_time = total_time / stats.items_processed as u32;
                    }

                    stats.last_update = Some(start);
                }

                QueueResult::Success(item)
            }
            None => QueueResult::Empty,
        }
    }

    /// Get current queue size without blocking
    pub fn size_non_blocking(&self) -> Option<usize> {
        self.queue.try_read().ok().map(|q| q.len())
    }

    /// Check if queue is empty without blocking
    pub fn is_empty_non_blocking(&self) -> Option<bool> {
        self.queue.try_read().ok().map(|q| q.is_empty())
    }

    /// Get queue statistics
    pub async fn get_stats(&self) -> QueueStats {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => QueueStats::default(), // Fallback if blocked
        }
    }

    /// Clear queue for UI reset operations
    pub async fn clear_non_blocking(&self) -> QueueResult<usize> {
        let mut queue_lock = match timeout(self.ui_timeout, self.queue.write()).await {
            Ok(lock) => lock,
            Err(_) => return QueueResult::Timeout,
        };

        let cleared_count = queue_lock.len();
        queue_lock.clear();
        drop(queue_lock);

        // Update statistics
        if let Ok(mut stats) = self.stats.try_write() {
            stats.current_size = 0;
            stats.last_update = Some(Instant::now());
        }

        QueueResult::Success(cleared_count)
    }

    /// Wait for items without blocking UI (with timeout)
    pub async fn wait_for_items(&self) -> Result<(), FeedbackError> {
        // Use notification system with timeout
        match timeout(self.ui_timeout, self.notify.notified()).await {
            Ok(_) => Ok(()),
            Err(_) => {
                // Update timeout statistics
                if let Ok(mut stats) = self.stats.try_write() {
                    stats.timeouts += 1;
                }
                Err(FeedbackError::Timeout)
            }
        }
    }

    /// Batch process items for better UI responsiveness
    pub async fn process_batch<F, R>(&self, batch_size: usize, processor: F) -> Vec<R>
    where
        F: Fn(T) -> R,
    {
        let mut results = Vec::new();
        let start = Instant::now();

        for _ in 0..batch_size {
            // Check if we've exceeded UI responsiveness threshold
            if start.elapsed() > self.ui_timeout {
                break;
            }

            match self.dequeue_non_blocking().await {
                QueueResult::Success(item) => {
                    results.push(processor(item));
                }
                QueueResult::Empty | QueueResult::Timeout => break,
                QueueResult::AtCapacity => break,
            }
        }

        results
    }
}

/// UI-responsive feedback processing queue
pub type FeedbackQueue<T> = UiResponsiveQueue<T>;

/// Create a standard feedback processing queue
pub fn create_feedback_queue<T>() -> FeedbackQueue<T>
where
    T: Send + Sync + 'static,
{
    UiResponsiveQueue::new(1000) // Default max size of 1000 items
}

/// Create a high-priority UI queue with lower latency
pub fn create_ui_priority_queue<T>() -> FeedbackQueue<T>
where
    T: Send + Sync + 'static,
{
    UiResponsiveQueue::with_ui_timeout(100, Duration::from_millis(10)) // 10ms timeout for UI
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_ui_responsive_queue_basic_operations() {
        let queue: UiResponsiveQueue<i32> = UiResponsiveQueue::new(10);

        // Test enqueue
        match queue.enqueue_non_blocking(42).await {
            QueueResult::Success(_) => {}
            _ => panic!("Enqueue should succeed"),
        }

        // Test size
        assert_eq!(queue.size_non_blocking(), Some(1));
        assert_eq!(queue.is_empty_non_blocking(), Some(false));

        // Test dequeue
        match queue.dequeue_non_blocking().await {
            QueueResult::Success(value) => assert_eq!(value, 42),
            _ => panic!("Dequeue should succeed"),
        }

        // Test empty
        assert_eq!(queue.is_empty_non_blocking(), Some(true));
    }

    #[tokio::test]
    async fn test_queue_capacity_limits() {
        let queue: UiResponsiveQueue<i32> = UiResponsiveQueue::new(2);

        // Fill queue to capacity
        assert!(matches!(
            queue.enqueue_non_blocking(1).await,
            QueueResult::Success(_)
        ));
        assert!(matches!(
            queue.enqueue_non_blocking(2).await,
            QueueResult::Success(_)
        ));

        // Should reject additional items
        assert!(matches!(
            queue.enqueue_non_blocking(3).await,
            QueueResult::AtCapacity
        ));
    }

    #[tokio::test]
    async fn test_ui_timeout_behavior() {
        let queue: UiResponsiveQueue<i32> =
            UiResponsiveQueue::with_ui_timeout(10, Duration::from_millis(1));

        // Add some items
        let _ = queue.enqueue_non_blocking(1).await;
        let _ = queue.enqueue_non_blocking(2).await;

        // Simulate heavy processing that might block
        sleep(Duration::from_millis(5)).await;

        // Operations should still be responsive
        match queue.dequeue_non_blocking().await {
            QueueResult::Success(_) | QueueResult::Timeout => {
                // Both are acceptable - success or timeout to preserve UI responsiveness
            }
            _ => panic!("Should either succeed or timeout for UI responsiveness"),
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let queue: UiResponsiveQueue<i32> = UiResponsiveQueue::new(10);

        // Add multiple items
        for i in 1..=5 {
            let _ = queue.enqueue_non_blocking(i).await;
        }

        // Process batch
        let results = queue.process_batch(3, |x| x * 2).await;
        assert!(results.len() <= 3);

        for result in results {
            assert!(result % 2 == 0); // All should be even (doubled)
        }
    }

    #[tokio::test]
    async fn test_queue_statistics() {
        let queue: UiResponsiveQueue<i32> = UiResponsiveQueue::new(10);

        // Add and remove items
        let _ = queue.enqueue_non_blocking(1).await;
        let _ = queue.enqueue_non_blocking(2).await;
        let _ = queue.dequeue_non_blocking().await;

        let stats = queue.get_stats().await;
        assert_eq!(stats.current_size, 1);
        assert!(stats.items_processed > 0);
        assert!(stats.last_update.is_some());
    }
}
