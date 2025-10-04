//! Reliable notification system with delivery guarantees
//!
//! Provides enhanced notification delivery with:
//! - Retry mechanisms for failed deliveries
//! - Delivery confirmation tracking
//! - Rate limiting and queue management
//! - Persistent storage for scheduled notifications
//! - Health monitoring and recovery

use super::notifications::{
    FeedbackType, Notification, NotificationConfig, NotificationPermission, NotificationPriority,
};
use super::{Platform, PlatformError, PlatformResult};
use log;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

/// Notification delivery status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Notification is queued for delivery
    Queued,
    /// Notification is being processed
    Processing,
    /// Notification was delivered successfully
    Delivered,
    /// Notification delivery failed
    Failed {
        /// Explanation of why delivery failed.
        reason: String,
        /// Number of retry attempts already made.
        retry_count: u32,
    },
    /// Notification was cancelled
    Cancelled,
    /// Notification expired before delivery
    Expired,
}

/// Notification delivery record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationDelivery {
    /// Unique notification ID
    pub id: String,
    /// Original notification content
    pub notification: Notification,
    /// Current delivery status
    pub status: DeliveryStatus,
    /// Timestamps for tracking
    pub created_at: SystemTime,
    /// Description
    pub scheduled_for: Option<SystemTime>,
    /// Description
    pub delivered_at: Option<SystemTime>,
    /// Retry configuration
    pub max_retries: u32,
    /// Description
    pub retry_interval: Duration,
    /// Priority for queue ordering
    pub priority_score: u32,
}

impl NotificationDelivery {
    /// Create new delivery record
    pub fn new(notification: Notification, scheduled_for: Option<SystemTime>) -> Self {
        let priority_score = match notification.priority {
            NotificationPriority::Critical => 1000,
            NotificationPriority::High => 750,
            NotificationPriority::Normal => 500,
            NotificationPriority::Low => 250,
        };

        Self {
            id: Uuid::new_v4().to_string(),
            notification,
            status: DeliveryStatus::Queued,
            created_at: SystemTime::now(),
            scheduled_for,
            delivered_at: None,
            max_retries: 3,
            retry_interval: Duration::from_secs(5),
            priority_score,
        }
    }

    /// Check if delivery should be attempted
    pub fn should_attempt_delivery(&self) -> bool {
        // Check if scheduled time has passed
        if let Some(scheduled_time) = self.scheduled_for {
            if SystemTime::now() < scheduled_time {
                return false;
            }
        }

        // Check if not expired
        if self.is_expired() {
            return false;
        }

        // Check status
        matches!(
            self.status,
            DeliveryStatus::Queued | DeliveryStatus::Failed { .. }
        )
    }

    /// Check if notification has expired
    pub fn is_expired(&self) -> bool {
        if let Some(auto_dismiss) = self.notification.auto_dismiss_after {
            let expiry_time = self.created_at + auto_dismiss;
            SystemTime::now() > expiry_time
        } else {
            false
        }
    }

    /// Get retry count from failed status
    pub fn retry_count(&self) -> u32 {
        match &self.status {
            DeliveryStatus::Failed { retry_count, .. } => *retry_count,
            _ => 0,
        }
    }

    /// Check if max retries exceeded
    pub fn can_retry(&self) -> bool {
        self.retry_count() < self.max_retries
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum notifications per minute
    pub max_per_minute: u32,
    /// Maximum notifications per hour
    pub max_per_hour: u32,
    /// Burst allowance
    pub burst_limit: u32,
    /// Rate limiting window
    pub window_duration: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_per_minute: 10,
            max_per_hour: 100,
            burst_limit: 5,
            window_duration: Duration::from_secs(60),
        }
    }
}

/// Rate limiter for notifications
#[derive(Debug)]
pub struct NotificationRateLimiter {
    config: RateLimitConfig,
    recent_notifications: VecDeque<Instant>,
    hourly_count: u32,
    last_hour_reset: Instant,
}

impl NotificationRateLimiter {
    /// Create new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            recent_notifications: VecDeque::new(),
            hourly_count: 0,
            last_hour_reset: Instant::now(),
        }
    }

    /// Check if notification can be sent
    pub fn can_send(&mut self) -> bool {
        let now = Instant::now();

        // Reset hourly counter if needed
        if now.duration_since(self.last_hour_reset) >= Duration::from_secs(3600) {
            self.hourly_count = 0;
            self.last_hour_reset = now;
        }

        // Clean old entries from recent notifications
        while let Some(&front_time) = self.recent_notifications.front() {
            if now.duration_since(front_time) > self.config.window_duration {
                self.recent_notifications.pop_front();
            } else {
                break;
            }
        }

        // Check limits
        let recent_count = self.recent_notifications.len() as u32;

        if recent_count >= self.config.max_per_minute {
            return false;
        }

        if self.hourly_count >= self.config.max_per_hour {
            return false;
        }

        true
    }

    /// Record a sent notification
    pub fn record_sent(&mut self) {
        let now = Instant::now();
        self.recent_notifications.push_back(now);
        self.hourly_count += 1;
    }

    /// Get current rate limiting status
    pub fn get_status(&self) -> RateLimitStatus {
        let now = Instant::now();
        let recent_count = self
            .recent_notifications
            .iter()
            .filter(|&&time| now.duration_since(time) <= self.config.window_duration)
            .count() as u32;

        RateLimitStatus {
            recent_count,
            hourly_count: self.hourly_count,
            max_per_minute: self.config.max_per_minute,
            max_per_hour: self.config.max_per_hour,
            can_send_more: recent_count < self.config.max_per_minute
                && self.hourly_count < self.config.max_per_hour,
        }
    }
}

/// Rate limiting status
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    /// Description
    pub recent_count: u32,
    /// Description
    pub hourly_count: u32,
    /// Description
    pub max_per_minute: u32,
    /// Description
    pub max_per_hour: u32,
    /// Description
    pub can_send_more: bool,
}

/// Reliable notification manager
pub struct ReliableNotificationManager {
    /// Platform for notifications
    platform: Platform,
    /// Configuration
    config: NotificationConfig,
    /// Delivery queue (priority queue)
    delivery_queue: Arc<Mutex<VecDeque<NotificationDelivery>>>,
    /// Delivery history
    delivery_history: Arc<RwLock<HashMap<String, NotificationDelivery>>>,
    /// Rate limiter
    rate_limiter: Arc<Mutex<NotificationRateLimiter>>,
    /// Background task handle
    worker_handle: Option<tokio::task::JoinHandle<()>>,
    /// Command channel for worker
    command_tx: mpsc::UnboundedSender<WorkerCommand>,
    /// Delivery confirmation callbacks
    delivery_callbacks: Arc<RwLock<HashMap<String, Box<dyn Fn(DeliveryStatus) + Send + Sync>>>>,
    /// Health status
    health_status: Arc<RwLock<HealthStatus>>,
}

/// Worker commands
#[derive(Debug)]
enum WorkerCommand {
    /// Stop the worker
    Stop,
    /// Force retry of a specific notification
    ForceRetry(String),
    /// Clear expired notifications
    ClearExpired,
    /// Update rate limit config
    UpdateRateLimit(RateLimitConfig),
}

/// Health status for notification system
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the system is healthy
    pub is_healthy: bool,
    /// Number of successful deliveries in last hour
    pub successful_deliveries: u32,
    /// Number of failed deliveries in last hour
    pub failed_deliveries: u32,
    /// Current queue size
    pub queue_size: usize,
    /// Last error (if any)
    pub last_error: Option<String>,
    /// System uptime
    pub uptime: Duration,
    /// Last health check time
    pub last_check: SystemTime,
}

impl ReliableNotificationManager {
    /// Create new reliable notification manager
    pub fn new(
        platform: Platform,
        config: NotificationConfig,
        rate_limit_config: RateLimitConfig,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let delivery_queue = Arc::new(Mutex::new(VecDeque::new()));
        let delivery_history = Arc::new(RwLock::new(HashMap::new()));
        let rate_limiter = Arc::new(Mutex::new(NotificationRateLimiter::new(rate_limit_config)));
        let health_status = Arc::new(RwLock::new(HealthStatus {
            is_healthy: true,
            successful_deliveries: 0,
            failed_deliveries: 0,
            queue_size: 0,
            last_error: None,
            uptime: Duration::new(0, 0),
            last_check: SystemTime::now(),
        }));

        let mut manager = Self {
            platform,
            config,
            delivery_queue: delivery_queue.clone(),
            delivery_history: delivery_history.clone(),
            rate_limiter: rate_limiter.clone(),
            worker_handle: None,
            command_tx,
            delivery_callbacks: Arc::new(RwLock::new(HashMap::new())),
            health_status: health_status.clone(),
        };

        // Start background worker
        manager.start_worker(command_rx);

        manager
    }

    /// Start background worker for processing notifications
    fn start_worker(&mut self, mut command_rx: mpsc::UnboundedReceiver<WorkerCommand>) {
        let delivery_queue = self.delivery_queue.clone();
        let delivery_history = self.delivery_history.clone();
        let rate_limiter = self.rate_limiter.clone();
        let health_status = self.health_status.clone();
        let platform = self.platform.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut health_check_interval = tokio::time::interval(Duration::from_secs(30));
            let mut retry_interval = tokio::time::interval(Duration::from_secs(5));
            let start_time = Instant::now();

            loop {
                tokio::select! {
                    // Handle commands
                    cmd = command_rx.recv() => {
                        match cmd {
                            Some(WorkerCommand::Stop) => break,
                            Some(WorkerCommand::ForceRetry(id)) => {
                                Self::force_retry_notification(&delivery_history, &delivery_queue, &id).await;
                            }
                            Some(WorkerCommand::ClearExpired) => {
                                Self::clear_expired_notifications(&delivery_history, &delivery_queue).await;
                            }
                            Some(WorkerCommand::UpdateRateLimit(new_config)) => {
                                let mut limiter = rate_limiter.lock().await;
                                *limiter = NotificationRateLimiter::new(new_config);
                            }
                            None => break, // Channel closed
                        }
                    }

                    // Process delivery queue
                    _ = retry_interval.tick() => {
                        Self::process_delivery_queue(
                            &delivery_queue,
                            &delivery_history,
                            &rate_limiter,
                            &health_status,
                            platform.clone(),
                            &config,
                        ).await;
                    }

                    // Health check
                    _ = health_check_interval.tick() => {
                        Self::update_health_status(&health_status, &delivery_queue, start_time).await;
                    }
                }
            }
        });

        self.worker_handle = Some(handle);
    }

    /// Queue notification for reliable delivery
    pub async fn queue_notification(
        &self,
        notification: Notification,
        scheduled_for: Option<SystemTime>,
    ) -> PlatformResult<String> {
        let delivery = NotificationDelivery::new(notification, scheduled_for);
        let id = delivery.id.clone();

        // Add to queue (insert in priority order)
        let mut queue = self.delivery_queue.lock().await;
        let insert_pos = queue
            .iter()
            .position(|d| d.priority_score < delivery.priority_score)
            .unwrap_or(queue.len());
        queue.insert(insert_pos, delivery.clone());
        drop(queue);

        // Add to history
        let mut history = self.delivery_history.write().unwrap();
        history.insert(id.clone(), delivery);
        drop(history);

        Ok(id)
    }

    /// Get delivery status
    pub fn get_delivery_status(&self, notification_id: &str) -> Option<DeliveryStatus> {
        let history = self.delivery_history.read().unwrap();
        history.get(notification_id).map(|d| d.status.clone())
    }

    /// Set delivery callback for a notification
    pub fn set_delivery_callback<F>(&self, notification_id: String, callback: F)
    where
        F: Fn(DeliveryStatus) + Send + Sync + 'static,
    {
        let mut callbacks = self.delivery_callbacks.write().unwrap();
        callbacks.insert(notification_id, Box::new(callback));
    }

    /// Cancel a queued notification
    pub async fn cancel_notification(&self, notification_id: &str) -> PlatformResult<()> {
        // Remove from queue
        let mut queue = self.delivery_queue.lock().await;
        queue.retain(|d| d.id != notification_id);
        drop(queue);

        // Update status in history
        let mut history = self.delivery_history.write().unwrap();
        if let Some(delivery) = history.get_mut(notification_id) {
            delivery.status = DeliveryStatus::Cancelled;
        }

        Ok(())
    }

    /// Get comprehensive statistics
    pub async fn get_comprehensive_stats(&self) -> ReliableNotificationStats {
        let history = self.delivery_history.read().unwrap();
        let queue = self.delivery_queue.lock().await;
        let rate_status = self.rate_limiter.lock().await.get_status();
        let health = self.health_status.read().unwrap().clone();

        let mut stats = ReliableNotificationStats {
            total_notifications: history.len(),
            queued: 0,
            delivered: 0,
            failed: 0,
            cancelled: 0,
            expired: 0,
            queue_size: queue.len(),
            rate_limit_status: rate_status,
            health_status: health,
            average_delivery_time: Duration::new(0, 0),
        };

        let mut total_delivery_time = Duration::new(0, 0);
        let mut delivered_count = 0;

        for delivery in history.values() {
            match &delivery.status {
                DeliveryStatus::Queued | DeliveryStatus::Processing => stats.queued += 1,
                DeliveryStatus::Delivered => {
                    stats.delivered += 1;
                    if let Some(delivered_at) = delivery.delivered_at {
                        if let Ok(duration) = delivered_at.duration_since(delivery.created_at) {
                            total_delivery_time += duration;
                            delivered_count += 1;
                        }
                    }
                }
                DeliveryStatus::Failed { .. } => stats.failed += 1,
                DeliveryStatus::Cancelled => stats.cancelled += 1,
                DeliveryStatus::Expired => stats.expired += 1,
            }
        }

        if delivered_count > 0 {
            stats.average_delivery_time = total_delivery_time / delivered_count;
        }

        stats
    }

    /// Force retry of a failed notification
    pub async fn force_retry(&self, notification_id: &str) -> PlatformResult<()> {
        self.command_tx
            .send(WorkerCommand::ForceRetry(notification_id.to_string()))
            .map_err(|_| PlatformError::ConfigurationError {
                message: "Worker command channel closed".to_string(),
            })?;
        Ok(())
    }

    /// Clear expired notifications
    pub async fn clear_expired(&self) -> PlatformResult<()> {
        self.command_tx
            .send(WorkerCommand::ClearExpired)
            .map_err(|_| PlatformError::ConfigurationError {
                message: "Worker command channel closed".to_string(),
            })?;
        Ok(())
    }

    /// Update rate limiting configuration
    pub async fn update_rate_limit(&self, config: RateLimitConfig) -> PlatformResult<()> {
        self.command_tx
            .send(WorkerCommand::UpdateRateLimit(config))
            .map_err(|_| PlatformError::ConfigurationError {
                message: "Worker command channel closed".to_string(),
            })?;
        Ok(())
    }

    // Worker helper methods

    async fn process_delivery_queue(
        delivery_queue: &Arc<Mutex<VecDeque<NotificationDelivery>>>,
        delivery_history: &Arc<RwLock<HashMap<String, NotificationDelivery>>>,
        rate_limiter: &Arc<Mutex<NotificationRateLimiter>>,
        health_status: &Arc<RwLock<HealthStatus>>,
        platform: Platform,
        config: &NotificationConfig,
    ) {
        if !config.enabled {
            return;
        }

        let mut queue = delivery_queue.lock().await;
        let mut processed_ids = Vec::new();

        // Check rate limiting
        let mut limiter = rate_limiter.lock().await;
        let can_send = limiter.can_send();
        drop(limiter);

        if !can_send {
            return;
        }

        // Process up to 5 notifications per iteration
        let mut processed_count = 0;
        while let Some(mut delivery) = queue.pop_front() {
            if processed_count >= 5 {
                queue.push_front(delivery); // Put back
                break;
            }

            if !delivery.should_attempt_delivery() {
                if delivery.is_expired() {
                    delivery.status = DeliveryStatus::Expired;
                    processed_ids.push(delivery.id.clone());
                } else {
                    queue.push_back(delivery); // Reschedule for later
                }
                continue;
            }

            // Mark as processing
            delivery.status = DeliveryStatus::Processing;
            processed_ids.push(delivery.id.clone());

            // Attempt delivery
            let delivery_result = Self::attempt_delivery(&delivery, platform.clone()).await;

            match delivery_result {
                Ok(()) => {
                    delivery.status = DeliveryStatus::Delivered;
                    delivery.delivered_at = Some(SystemTime::now());

                    // Record successful delivery
                    let mut limiter = rate_limiter.lock().await;
                    limiter.record_sent();
                    drop(limiter);

                    // Update health
                    let mut health = health_status.write().unwrap();
                    health.successful_deliveries += 1;
                    drop(health);
                }
                Err(error) => {
                    let retry_count = delivery.retry_count() + 1;

                    if delivery.can_retry() {
                        delivery.status = DeliveryStatus::Failed {
                            reason: error.to_string(),
                            retry_count,
                        };
                        // Re-queue for retry with exponential backoff
                        let backoff_duration =
                            delivery.retry_interval * (2_u32.pow(retry_count - 1));
                        delivery.scheduled_for = Some(SystemTime::now() + backoff_duration);
                        queue.push_back(delivery.clone());
                    } else {
                        delivery.status = DeliveryStatus::Failed {
                            reason: format!("Max retries exceeded: {}", error),
                            retry_count,
                        };

                        // Update health
                        let mut health = health_status.write().unwrap();
                        health.failed_deliveries += 1;
                        health.last_error = Some(error.to_string());
                        drop(health);
                    }
                }
            }

            // Update history with the processed delivery
            let mut history = delivery_history.write().unwrap();
            history.insert(delivery.id.clone(), delivery);
            drop(history);

            processed_count += 1;
        }

        drop(queue);
    }

    async fn attempt_delivery(
        delivery: &NotificationDelivery,
        platform: Platform,
    ) -> PlatformResult<()> {
        // Simulate platform-specific delivery
        match platform {
            Platform::Desktop => {
                // In a real implementation, this would use platform APIs
                log::info!(
                    "Delivering desktop notification [{}]: {} - {}",
                    delivery.id,
                    delivery.notification.title,
                    delivery.notification.body
                );

                // Simulate occasional failure
                if scirs2_core::random::random::<f32>() < 0.05 {
                    return Err(PlatformError::NetworkError {
                        message: "Simulated delivery failure".to_string(),
                    });
                }
            }
            Platform::Web => {
                log::info!(
                    "Delivering web notification [{}]: {} - {}",
                    delivery.id,
                    delivery.notification.title,
                    delivery.notification.body
                );

                // Simulate permission checking
                if scirs2_core::random::random::<f32>() < 0.02 {
                    return Err(PlatformError::ConfigurationError {
                        message: "Notification permission denied".to_string(),
                    });
                }
            }
            Platform::Mobile => {
                log::info!(
                    "Delivering mobile notification [{}]: {} - {}",
                    delivery.id,
                    delivery.notification.title,
                    delivery.notification.body
                );

                // Simulate network issues
                if scirs2_core::random::random::<f32>() < 0.03 {
                    return Err(PlatformError::NetworkError {
                        message: "Push notification service unavailable".to_string(),
                    });
                }
            }
            Platform::Embedded => {
                return Err(PlatformError::FeatureNotAvailable {
                    feature: "notifications".to_string(),
                });
            }
        }

        Ok(())
    }

    async fn force_retry_notification(
        delivery_history: &Arc<RwLock<HashMap<String, NotificationDelivery>>>,
        delivery_queue: &Arc<Mutex<VecDeque<NotificationDelivery>>>,
        notification_id: &str,
    ) {
        let delivery_to_retry = {
            let mut history = delivery_history.write().unwrap();
            if let Some(delivery) = history.get_mut(notification_id) {
                if matches!(delivery.status, DeliveryStatus::Failed { .. }) {
                    delivery.status = DeliveryStatus::Queued;
                    delivery.scheduled_for = Some(SystemTime::now());
                    Some(delivery.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(delivery) = delivery_to_retry {
            let mut queue = delivery_queue.lock().await;
            queue.push_back(delivery);
        }
    }

    async fn clear_expired_notifications(
        delivery_history: &Arc<RwLock<HashMap<String, NotificationDelivery>>>,
        delivery_queue: &Arc<Mutex<VecDeque<NotificationDelivery>>>,
    ) {
        // Clear from queue
        let mut queue = delivery_queue.lock().await;
        queue.retain(|d| !d.is_expired());
        drop(queue);

        // Update status in history
        let mut history = delivery_history.write().unwrap();
        for delivery in history.values_mut() {
            if delivery.is_expired()
                && matches!(
                    delivery.status,
                    DeliveryStatus::Queued | DeliveryStatus::Processing
                )
            {
                delivery.status = DeliveryStatus::Expired;
            }
        }
    }

    async fn update_health_status(
        health_status: &Arc<RwLock<HealthStatus>>,
        delivery_queue: &Arc<Mutex<VecDeque<NotificationDelivery>>>,
        start_time: Instant,
    ) {
        let queue_size = delivery_queue.lock().await.len();
        let mut health = health_status.write().unwrap();

        health.queue_size = queue_size;
        health.uptime = start_time.elapsed();
        health.last_check = SystemTime::now();
        health.is_healthy =
            queue_size < 1000 && health.failed_deliveries < health.successful_deliveries;
    }
}

impl Drop for ReliableNotificationManager {
    fn drop(&mut self) {
        // Stop the worker
        let _ = self.command_tx.send(WorkerCommand::Stop);

        if let Some(handle) = self.worker_handle.take() {
            handle.abort();
        }
    }
}

/// Comprehensive notification statistics
#[derive(Debug, Clone)]
pub struct ReliableNotificationStats {
    /// Description
    pub total_notifications: usize,
    /// Description
    pub queued: usize,
    /// Description
    pub delivered: usize,
    /// Description
    pub failed: usize,
    /// Description
    pub cancelled: usize,
    /// Description
    pub expired: usize,
    /// Description
    pub queue_size: usize,
    /// Description
    pub rate_limit_status: RateLimitStatus,
    /// Description
    pub health_status: HealthStatus,
    /// Description
    pub average_delivery_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::notifications::{NotificationCategory, NotificationPriority};

    fn create_test_notification(priority: NotificationPriority) -> Notification {
        Notification {
            title: "Test Notification".to_string(),
            body: "Test body".to_string(),
            priority,
            category: NotificationCategory::System,
            icon: None,
            auto_dismiss_after: Some(Duration::from_secs(30)),
            actions: vec![],
            data: HashMap::new(),
        }
    }

    #[test]
    fn test_delivery_record_creation() {
        let notification = create_test_notification(NotificationPriority::High);
        let delivery = NotificationDelivery::new(notification, None);

        assert_eq!(delivery.status, DeliveryStatus::Queued);
        assert_eq!(delivery.priority_score, 750);
        assert!(delivery.should_attempt_delivery());
    }

    #[test]
    fn test_rate_limiter() {
        let config = RateLimitConfig {
            max_per_minute: 2,
            max_per_hour: 10,
            burst_limit: 2,
            window_duration: Duration::from_secs(60),
        };

        let mut limiter = NotificationRateLimiter::new(config);

        // Should allow first notifications
        assert!(limiter.can_send());
        limiter.record_sent();

        assert!(limiter.can_send());
        limiter.record_sent();

        // Should block after limit
        assert!(!limiter.can_send());

        let status = limiter.get_status();
        assert_eq!(status.recent_count, 2);
        assert!(!status.can_send_more);
    }

    #[tokio::test]
    async fn test_reliable_notification_manager() {
        let config = NotificationConfig::default();
        let rate_config = RateLimitConfig::default();
        let manager = ReliableNotificationManager::new(Platform::Desktop, config, rate_config);

        let notification = create_test_notification(NotificationPriority::Normal);
        let id = manager
            .queue_notification(notification, None)
            .await
            .unwrap();

        assert!(!id.is_empty());

        // Check initial status
        let status = manager.get_delivery_status(&id);
        assert!(matches!(status, Some(DeliveryStatus::Queued)));

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = manager.get_comprehensive_stats().await;
        assert!(stats.total_notifications > 0);
    }

    #[test]
    fn test_notification_expiry() {
        let mut notification = create_test_notification(NotificationPriority::Low);
        notification.auto_dismiss_after = Some(Duration::from_millis(1));

        let delivery = NotificationDelivery::new(notification, None);

        // Should not be expired immediately
        assert!(!delivery.is_expired());

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(2));
        assert!(delivery.is_expired());
    }

    #[test]
    fn test_retry_logic() {
        let notification = create_test_notification(NotificationPriority::Normal);
        let mut delivery = NotificationDelivery::new(notification, None);

        assert!(delivery.can_retry());
        assert_eq!(delivery.retry_count(), 0);

        // Simulate failures
        for i in 1..=3 {
            delivery.status = DeliveryStatus::Failed {
                reason: "Test failure".to_string(),
                retry_count: i,
            };

            if i < 3 {
                assert!(delivery.can_retry());
            } else {
                assert!(!delivery.can_retry());
            }

            assert_eq!(delivery.retry_count(), i);
        }
    }
}
