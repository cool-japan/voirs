//! Enhanced notification reliability system
//!
//! This module provides enhanced reliability mechanisms for the notification system
//! to ensure consistent delivery and proper error handling.

use super::notifications::{Notification, NotificationPriority};
use super::{Platform, PlatformError, PlatformResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{interval, sleep_until, timeout, MissedTickBehavior};
use uuid::Uuid;

// Helper for serializing Instant as duration since a reference point
fn serialize_instant<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let duration = instant.elapsed();
    duration.serialize(serializer)
}

fn deserialize_instant<'de, D>(deserializer: D) -> Result<Instant, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let duration = Duration::deserialize(deserializer)?;
    Ok(Instant::now() - duration)
}

/// Enhanced delivery status with more granular states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnhancedDeliveryStatus {
    /// Queued for delivery
    Queued { priority_score: u32 },
    /// Currently being processed
    Processing {
        #[serde(
            serialize_with = "serialize_instant",
            deserialize_with = "deserialize_instant"
        )]
        started_at: Instant,
    },
    /// Successfully delivered
    Delivered {
        #[serde(
            serialize_with = "serialize_instant",
            deserialize_with = "deserialize_instant"
        )]
        delivered_at: Instant,
        attempts: u32,
    },
    /// Failed delivery with retry schedule
    Failed {
        reason: String,
        #[serde(
            serialize_with = "serialize_option_instant",
            deserialize_with = "deserialize_option_instant"
        )]
        next_retry: Option<Instant>,
        attempt: u32,
    },
    /// Permanently failed (max retries exceeded)
    Abandoned { reason: String, final_attempt: u32 },
    /// Cancelled by user or system
    Cancelled {
        #[serde(
            serialize_with = "serialize_instant",
            deserialize_with = "deserialize_instant"
        )]
        cancelled_at: Instant,
    },
    /// Expired before delivery
    Expired {
        #[serde(
            serialize_with = "serialize_instant",
            deserialize_with = "deserialize_instant"
        )]
        expired_at: Instant,
    },
}

// Helper for serializing Option<Instant>
fn serialize_option_instant<S>(instant: &Option<Instant>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match instant {
        Some(i) => {
            let duration = i.elapsed();
            Some(duration).serialize(serializer)
        }
        None => None::<Duration>.serialize(serializer),
    }
}

fn deserialize_option_instant<'de, D>(deserializer: D) -> Result<Option<Instant>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let duration_opt = Option::<Duration>::deserialize(deserializer)?;
    Ok(duration_opt.map(|d| Instant::now() - d))
}

/// Comprehensive notification record with reliability tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliableNotification {
    /// Unique notification ID
    pub id: String,
    /// Original notification content
    pub notification: Notification,
    /// Enhanced delivery status
    pub status: EnhancedDeliveryStatus,
    /// Creation timestamp
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub created_at: Instant,
    /// Scheduled delivery time
    #[serde(
        serialize_with = "serialize_option_instant",
        deserialize_with = "deserialize_option_instant"
    )]
    pub scheduled_for: Option<Instant>,
    /// Retry configuration
    pub max_retries: u32,
    pub retry_backoff: Duration,
    /// Priority score for queue ordering
    pub priority_score: u32,
    /// Error history for debugging
    pub error_history: Vec<NotificationError>,
    /// Delivery metadata
    pub metadata: HashMap<String, String>,
}

impl ReliableNotification {
    /// Create a new reliable notification
    pub fn new(notification: Notification, scheduled_for: Option<Instant>) -> Self {
        let priority_score = match notification.priority {
            NotificationPriority::Critical => 1000,
            NotificationPriority::High => 750,
            NotificationPriority::Normal => 500,
            NotificationPriority::Low => 250,
        };

        let max_retries = match notification.priority {
            NotificationPriority::Critical => 5,
            NotificationPriority::High => 3,
            NotificationPriority::Normal => 2,
            NotificationPriority::Low => 1,
        };

        Self {
            id: Uuid::new_v4().to_string(),
            notification,
            status: EnhancedDeliveryStatus::Queued { priority_score },
            created_at: Instant::now(),
            scheduled_for,
            max_retries,
            retry_backoff: Duration::from_secs(2),
            priority_score,
            error_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if notification should be delivered now
    pub fn should_deliver_now(&self) -> bool {
        // Check if scheduled time has passed
        if let Some(scheduled_time) = self.scheduled_for {
            if Instant::now() < scheduled_time {
                return false;
            }
        }

        // Check if not expired (30 minutes max age)
        if self.created_at.elapsed() > Duration::from_secs(1800) {
            return false;
        }

        // Check status allows delivery
        match &self.status {
            EnhancedDeliveryStatus::Queued { .. } => true,
            EnhancedDeliveryStatus::Failed {
                next_retry: Some(next),
                ..
            } => Instant::now() >= *next,
            _ => false,
        }
    }

    /// Calculate next retry time with exponential backoff
    pub fn calculate_next_retry(&self, attempt: u32) -> Option<Instant> {
        if attempt >= self.max_retries {
            return None;
        }

        let backoff_multiplier = 2_u32.pow(attempt);
        let delay = self.retry_backoff * backoff_multiplier;
        Some(Instant::now() + delay)
    }

    /// Add error to history
    pub fn add_error(&mut self, error: NotificationError) {
        self.error_history.push(error);

        // Keep only last 10 errors to prevent memory bloat
        if self.error_history.len() > 10 {
            self.error_history.drain(0..self.error_history.len() - 10);
        }
    }
}

/// Notification error with debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationError {
    /// Error message
    pub message: String,
    /// Error type
    pub error_type: NotificationErrorType,
    /// Timestamp when error occurred
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub occurred_at: Instant,
    /// Attempt number when error occurred
    pub attempt: u32,
    /// Additional debugging context
    pub context: HashMap<String, String>,
}

/// Types of notification errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationErrorType {
    /// Platform not available
    PlatformUnavailable,
    /// Permission denied
    PermissionDenied,
    /// Rate limit exceeded
    RateLimited,
    /// Network connectivity issue
    NetworkError,
    /// Timeout during delivery
    TimeoutError,
    /// Invalid notification format
    InvalidFormat,
    /// System resource exhaustion
    ResourceExhausted,
    /// Unknown platform error
    UnknownError,
}

/// Enhanced notification reliability manager
pub struct NotificationReliabilityManager {
    /// Pending notifications by ID
    notifications: Arc<RwLock<HashMap<String, ReliableNotification>>>,
    /// Priority queue for delivery scheduling
    delivery_queue: Arc<Mutex<VecDeque<String>>>,
    /// Statistics and health monitoring
    stats: Arc<RwLock<ReliabilityStats>>,
    /// Configuration
    config: ReliabilityConfig,
    /// Health check status
    health_status: Arc<RwLock<HealthStatus>>,
}

/// Reliability statistics
#[derive(Debug, Clone, Default)]
pub struct ReliabilityStats {
    /// Total notifications processed
    pub total_processed: u64,
    /// Successful deliveries
    pub successful_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Abandoned notifications (max retries exceeded)
    pub abandoned_notifications: u64,
    /// Current queue size
    pub current_queue_size: usize,
    /// Average delivery time
    pub average_delivery_time: Duration,
    /// Error rate over last 100 notifications
    pub recent_error_rate: f32,
    /// Last cleanup time
    pub last_cleanup: Option<Instant>,
}

/// Health status of the notification system
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Overall system health
    pub overall_health: SystemHealth,
    /// Individual component health
    pub component_health: HashMap<String, ComponentHealth>,
    /// Last health check time
    pub last_check: Instant,
    /// Health check errors
    pub health_errors: Vec<String>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            overall_health: SystemHealth::Healthy,
            component_health: HashMap::new(),
            last_check: Instant::now(),
            health_errors: Vec::new(),
        }
    }
}

/// System health levels
#[derive(Debug, Clone, PartialEq)]
pub enum SystemHealth {
    Healthy,
    Degraded,
    Critical,
    Failed,
}

/// Component health information
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub status: SystemHealth,
    pub last_success: Option<Instant>,
    pub consecutive_failures: u32,
    pub response_time: Option<Duration>,
}

/// Reliability configuration
#[derive(Debug, Clone)]
pub struct ReliabilityConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum notification age before expiration
    pub max_notification_age: Duration,
    /// Enable persistent storage
    pub enable_persistence: bool,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(60), // 1 minute
            max_notification_age: Duration::from_secs(1800), // 30 minutes
            enable_persistence: false,
        }
    }
}

impl NotificationReliabilityManager {
    /// Create a new reliability manager
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            notifications: Arc::new(RwLock::new(HashMap::new())),
            delivery_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(ReliabilityStats::default())),
            config,
            health_status: Arc::new(RwLock::new(HealthStatus::default())),
        }
    }

    /// Add notification to reliable delivery queue
    pub async fn enqueue_notification(
        &self,
        notification: Notification,
        scheduled_for: Option<Instant>,
    ) -> PlatformResult<String> {
        let reliable_notif = ReliableNotification::new(notification, scheduled_for);
        let id = reliable_notif.id.clone();

        // Check queue capacity
        {
            let notifications = self.notifications.read().await;
            if notifications.len() >= self.config.max_queue_size {
                return Err(PlatformError::CapacityExceeded {
                    current: notifications.len(),
                    max: self.config.max_queue_size,
                });
            }
        }

        // Add to storage and queue
        {
            let mut notifications = self.notifications.write().await;
            notifications.insert(id.clone(), reliable_notif);
        }

        {
            let mut queue = self.delivery_queue.lock().await;
            queue.push_back(id.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.current_queue_size += 1;
        }

        log::debug!("Enqueued notification {} for reliable delivery", id);
        Ok(id)
    }

    /// Get notification status
    pub async fn get_notification_status(&self, id: &str) -> Option<EnhancedDeliveryStatus> {
        let notifications = self.notifications.read().await;
        notifications.get(id).map(|n| n.status.clone())
    }

    /// Mark notification as delivered
    pub async fn mark_delivered(&self, id: &str, attempts: u32) -> PlatformResult<()> {
        let mut notifications = self.notifications.write().await;
        if let Some(notification) = notifications.get_mut(id) {
            notification.status = EnhancedDeliveryStatus::Delivered {
                delivered_at: Instant::now(),
                attempts,
            };

            // Update stats
            drop(notifications);
            let mut stats = self.stats.write().await;
            stats.successful_deliveries += 1;
            stats.total_processed += 1;

            log::debug!(
                "Marked notification {} as delivered after {} attempts",
                id,
                attempts
            );
            Ok(())
        } else {
            Err(PlatformError::ConfigurationError {
                message: format!("notification {} not found", id),
            })
        }
    }

    /// Mark notification as failed and schedule retry if appropriate
    pub async fn mark_failed(&self, id: &str, error: NotificationError) -> PlatformResult<bool> {
        let mut notifications = self.notifications.write().await;
        if let Some(notification) = notifications.get_mut(id) {
            let current_attempt = match &notification.status {
                EnhancedDeliveryStatus::Failed { attempt, .. } => attempt + 1,
                _ => 1,
            };

            notification.add_error(error.clone());

            let next_retry = notification.calculate_next_retry(current_attempt);
            let will_retry = next_retry.is_some();

            notification.status = if will_retry {
                EnhancedDeliveryStatus::Failed {
                    reason: error.message,
                    next_retry,
                    attempt: current_attempt,
                }
            } else {
                EnhancedDeliveryStatus::Abandoned {
                    reason: error.message,
                    final_attempt: current_attempt,
                }
            };

            // Update stats
            drop(notifications);
            let mut stats = self.stats.write().await;
            if will_retry {
                stats.failed_deliveries += 1;
            } else {
                stats.abandoned_notifications += 1;
            }
            stats.total_processed += 1;

            log::debug!(
                "Marked notification {} as {} (attempt {})",
                id,
                if will_retry {
                    "failed with retry"
                } else {
                    "abandoned"
                },
                current_attempt
            );

            Ok(will_retry)
        } else {
            Err(PlatformError::ConfigurationError {
                message: format!("notification {} not found", id),
            })
        }
    }

    /// Get next notification ready for delivery
    pub async fn get_next_for_delivery(&self) -> Option<String> {
        let notifications = self.notifications.read().await;
        let mut queue = self.delivery_queue.lock().await;

        // Find next deliverable notification
        while let Some(id) = queue.pop_front() {
            if let Some(notification) = notifications.get(&id) {
                if notification.should_deliver_now() {
                    queue.push_front(id.clone()); // Put it back for processing
                    return Some(id);
                }
                // If not ready, continue to next
            }
        }

        None
    }

    /// Perform cleanup of expired and completed notifications
    pub async fn cleanup(&self) -> PlatformResult<u32> {
        let start_time = Instant::now();
        let mut notifications = self.notifications.write().await;
        let mut queue = self.delivery_queue.lock().await;

        let initial_count = notifications.len();
        let max_age = self.config.max_notification_age;

        // Remove expired notifications
        notifications.retain(|_, notification| {
            let is_not_expired = notification.created_at.elapsed() <= max_age;
            let is_not_completed = !matches!(
                notification.status,
                EnhancedDeliveryStatus::Delivered { .. }
                    | EnhancedDeliveryStatus::Abandoned { .. }
                    | EnhancedDeliveryStatus::Cancelled { .. }
                    | EnhancedDeliveryStatus::Expired { .. }
            );
            is_not_expired && is_not_completed
        });

        // Clean up queue
        queue.retain(|id| notifications.contains_key(id));

        let final_count = notifications.len();
        let cleaned_count = initial_count - final_count;

        // Update stats
        drop(notifications);
        drop(queue);
        let mut stats = self.stats.write().await;
        stats.current_queue_size = final_count;
        stats.last_cleanup = Some(start_time);

        log::info!(
            "Cleaned up {} expired/completed notifications",
            cleaned_count
        );
        Ok(cleaned_count as u32)
    }

    /// Get reliability statistics
    pub async fn get_stats(&self) -> ReliabilityStats {
        self.stats.read().await.clone()
    }

    /// Get health status
    pub async fn get_health_status(&self) -> HealthStatus {
        self.health_status.read().await.clone()
    }

    /// Start background health monitoring
    pub async fn start_health_monitoring(&self) {
        let health_status = Arc::clone(&self.health_status);
        let notifications = Arc::clone(&self.notifications);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                // Perform health checks
                let mut health = health_status.write().await;
                health.last_check = Instant::now();
                health.health_errors.clear();

                // Check queue health
                let notifications_count = notifications.read().await.len();
                let queue_health = if notifications_count < config.max_queue_size / 2 {
                    SystemHealth::Healthy
                } else if notifications_count < (config.max_queue_size * 3) / 4 {
                    SystemHealth::Degraded
                } else {
                    SystemHealth::Critical
                };

                health.component_health.insert(
                    "queue".to_string(),
                    ComponentHealth {
                        status: queue_health.clone(),
                        last_success: Some(Instant::now()),
                        consecutive_failures: 0,
                        response_time: None,
                    },
                );

                // Check error rate
                let stats_read = stats.read().await;
                let error_rate_health = if stats_read.recent_error_rate < 0.1 {
                    SystemHealth::Healthy
                } else if stats_read.recent_error_rate < 0.25 {
                    SystemHealth::Degraded
                } else {
                    SystemHealth::Critical
                };

                health.component_health.insert(
                    "error_rate".to_string(),
                    ComponentHealth {
                        status: error_rate_health.clone(),
                        last_success: Some(Instant::now()),
                        consecutive_failures: 0,
                        response_time: None,
                    },
                );

                // Determine overall health
                health.overall_health = health
                    .component_health
                    .values()
                    .map(|c| &c.status)
                    .max_by_key(|status| match status {
                        SystemHealth::Healthy => 0,
                        SystemHealth::Degraded => 1,
                        SystemHealth::Critical => 2,
                        SystemHealth::Failed => 3,
                    })
                    .cloned()
                    .unwrap_or(SystemHealth::Healthy);

                drop(health);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reliable_notification_creation() {
        let notification = Notification {
            title: "Test".to_string(),
            body: "Test notification".to_string(),
            priority: NotificationPriority::High,
            category: super::super::notifications::NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: Vec::new(),
            data: HashMap::new(),
        };

        let reliable = ReliableNotification::new(notification.clone(), None);
        assert!(matches!(
            reliable.status,
            EnhancedDeliveryStatus::Queued { .. }
        ));
        assert_eq!(reliable.max_retries, 3); // High priority = 3 retries
        assert_eq!(reliable.notification.title, "Test");
    }

    #[tokio::test]
    async fn test_reliability_manager_enqueue() {
        let manager = NotificationReliabilityManager::new(ReliabilityConfig::default());

        let notification = Notification {
            title: "Test".to_string(),
            body: "Test notification".to_string(),
            priority: NotificationPriority::Normal,
            category: super::super::notifications::NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: Vec::new(),
            data: HashMap::new(),
        };

        let id = manager
            .enqueue_notification(notification, None)
            .await
            .unwrap();
        assert!(!id.is_empty());

        let status = manager.get_notification_status(&id).await;
        assert!(status.is_some());
        assert!(matches!(
            status.unwrap(),
            EnhancedDeliveryStatus::Queued { .. }
        ));
    }

    #[tokio::test]
    async fn test_notification_retry_logic() {
        let mut notification = ReliableNotification::new(
            Notification {
                title: "Test".to_string(),
                body: "Test notification".to_string(),
                priority: NotificationPriority::Critical,
                category: super::super::notifications::NotificationCategory::System,
                icon: None,
                auto_dismiss_after: None,
                actions: Vec::new(),
                data: HashMap::new(),
            },
            None,
        );

        // Test retry calculation
        let retry1 = notification.calculate_next_retry(1);
        assert!(retry1.is_some());

        let retry2 = notification.calculate_next_retry(2);
        assert!(retry2.is_some());

        // Should not retry after max attempts
        let retry_max = notification.calculate_next_retry(10);
        assert!(retry_max.is_none());
    }

    #[tokio::test]
    async fn test_cleanup() {
        let manager = NotificationReliabilityManager::new(ReliabilityConfig {
            max_notification_age: Duration::from_millis(100), // Very short for testing
            ..Default::default()
        });

        let notification = Notification {
            title: "Test".to_string(),
            body: "Test notification".to_string(),
            priority: NotificationPriority::Normal,
            category: super::super::notifications::NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: Vec::new(),
            data: HashMap::new(),
        };

        let id = manager
            .enqueue_notification(notification, None)
            .await
            .unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(200)).await;

        let cleaned = manager.cleanup().await.unwrap();
        assert!(cleaned > 0);

        // Notification should be gone
        let status = manager.get_notification_status(&id).await;
        assert!(status.is_none());
    }
}
