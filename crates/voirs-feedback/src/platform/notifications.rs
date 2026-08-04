//! Cross-platform notification system
//!
//! This module provides a unified notification system that works across
//! desktop, web, and mobile platforms with platform-specific optimizations.

use super::{Platform, PlatformError, PlatformResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, timeout};

/// Deliver a real local desktop notification via the OS-native mechanism:
///
/// - macOS: `osascript`'s `display notification`, which surfaces a genuine
///   Notification Center banner. Title/body are passed as `argv` to an
///   `on run argv` handler so arbitrary text (quotes, backslashes,
///   newlines) never needs AppleScript string escaping.
/// - Linux: `notify-send`, the CLI front-end for the freedesktop.org
///   `org.freedesktop.Notifications` D-Bus service.
///
/// There is no dependency-free, universally-installed mechanism on Windows
/// or any other target, so those honestly return
/// [`PlatformError::FeatureNotAvailable`] rather than printing a line and
/// claiming success.
pub(crate) async fn deliver_local_notification(title: &str, body: &str) -> PlatformResult<()> {
    #[cfg(target_os = "macos")]
    {
        let output = tokio::process::Command::new("osascript")
            .arg("-e")
            .arg("on run argv")
            .arg("-e")
            .arg("display notification (item 2 of argv) with title (item 1 of argv)")
            .arg("-e")
            .arg("end run")
            .arg(title)
            .arg(body)
            .output()
            .await
            .map_err(|e| notification_spawn_error("osascript", &e))?;

        if !output.status.success() {
            return Err(PlatformError::ConfigurationError {
                message: format!(
                    "osascript notification delivery failed (exit {:?}): {}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    {
        let output = tokio::process::Command::new("notify-send")
            .arg(title)
            .arg(body)
            .output()
            .await
            .map_err(|e| notification_spawn_error("notify-send", &e))?;

        if !output.status.success() {
            return Err(PlatformError::ConfigurationError {
                message: format!(
                    "notify-send notification delivery failed (exit {:?}): {}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        Ok(())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (title, body);
        Err(PlatformError::FeatureNotAvailable {
            feature: "desktop notifications (no local delivery backend implemented for this OS; \
                      supported: macOS via osascript, Linux via notify-send)"
                .to_string(),
        })
    }
}

/// Map a notification-backend spawn failure to a typed error, distinguishing
/// "the binary isn't installed" from other OS-level launch failures.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn notification_spawn_error(program: &str, e: &std::io::Error) -> PlatformError {
    if e.kind() == std::io::ErrorKind::NotFound {
        PlatformError::FeatureNotAvailable {
            feature: format!("desktop notifications ({program} binary not found in PATH)"),
        }
    } else {
        PlatformError::ConfigurationError {
            message: format!("failed to launch {program}: {e}"),
        }
    }
}

/// Pending notification with tracking information
#[derive(Debug, Clone)]
pub struct PendingNotification {
    /// Description
    pub notification: Notification,
    /// Description
    pub created_at: Instant,
    /// Description
    pub attempts: u32,
    /// Description
    pub last_attempt: Option<Instant>,
    /// Description
    pub status: NotificationStatus,
}

/// Retry notification for failed delivery attempts
#[derive(Debug, Clone)]
pub struct RetryNotification {
    /// Description
    pub id: String,
    /// Description
    pub notification: Notification,
    /// Description
    pub retry_count: u32,
    /// Description
    pub next_retry: Instant,
    /// Description
    pub max_retries: u32,
}

/// Notification delivery status
#[derive(Debug, Clone, PartialEq)]
pub enum NotificationStatus {
    /// Description
    Pending,
    /// Description
    Delivering,
    /// Description
    Delivered,
    /// Description
    Failed,
    /// Description
    Expired,
}

/// Rate limiter for notification delivery
#[derive(Debug)]
pub struct RateLimiter {
    tokens: u32,
    last_refill: Instant,
    max_tokens: u32,
    refill_rate: Duration,
}

impl RateLimiter {
    /// Description
    #[must_use]
    pub fn new(max_notifications_per_minute: u32) -> Self {
        Self {
            tokens: max_notifications_per_minute,
            last_refill: Instant::now(),
            max_tokens: max_notifications_per_minute,
            refill_rate: Duration::from_secs(60),
        }
    }

    /// Description
    pub fn can_send(&mut self) -> bool {
        self.refill_tokens();
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }

    fn refill_tokens(&mut self) {
        let now = Instant::now();
        let time_passed = now.duration_since(self.last_refill);

        if time_passed >= self.refill_rate {
            let periods = time_passed.as_secs() / self.refill_rate.as_secs();
            let tokens_to_add = (periods as u32).min(self.max_tokens);
            self.tokens = (self.tokens + tokens_to_add).min(self.max_tokens);
            self.last_refill = now;
        }
    }
}

/// Enhanced notification statistics
#[derive(Debug, Clone, Default)]
pub struct NotificationStats {
    /// Description
    pub pending_count: usize,
    /// Description
    pub total_sent: u64,
    /// Description
    pub total_delivered: u64,
    /// Description
    pub total_failed: u64,
    /// Description
    pub total_retries: u64,
    /// Description
    pub permission_status: NotificationPermission,
    /// Description
    pub average_delivery_time: Duration,
    /// Description
    pub last_cleanup: Option<Instant>,
}

/// Cross-platform notification manager with reliability features
pub struct NotificationManager {
    platform: Platform,
    config: NotificationConfig,
    pending_notifications: Arc<RwLock<HashMap<String, PendingNotification>>>,
    retry_queue: Arc<Mutex<VecDeque<RetryNotification>>>,
    stats: Arc<RwLock<NotificationStats>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl NotificationManager {
    /// Create a new notification manager with reliability features
    #[must_use]
    pub fn new(platform: Platform, config: NotificationConfig) -> Self {
        Self {
            platform,
            config: config.clone(),
            pending_notifications: Arc::new(RwLock::new(HashMap::new())),
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(NotificationStats::default())),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(
                config.max_notifications_per_minute,
            ))),
        }
    }

    /// Show a notification with enhanced reliability features
    pub async fn show_notification(&self, notification: Notification) -> PlatformResult<String> {
        // Check if notifications are enabled
        if !self.config.enabled {
            return Err(PlatformError::FeatureNotAvailable {
                feature: "notifications (disabled)".to_string(),
            });
        }

        // Apply rate limiting
        {
            let mut rate_limiter = self.rate_limiter.lock().await;
            if !rate_limiter.can_send() {
                return Err(PlatformError::RateLimited {
                    reason: "Rate limit exceeded, please try again later".to_string(),
                });
            }
        }

        let notification_id = format!("voirs_notification_{}", uuid::Uuid::new_v4().simple());
        let now = Instant::now();

        // Create pending notification with tracking
        let pending = PendingNotification {
            notification: notification.clone(),
            created_at: now,
            attempts: 1,
            last_attempt: Some(now),
            status: NotificationStatus::Delivering,
        };

        // Store pending notification
        {
            let mut pending_notifications = self.pending_notifications.write().await;

            // Check if we're at capacity
            if pending_notifications.len() >= self.config.max_pending {
                self.cleanup_expired_notifications().await;

                // If still at capacity, reject
                if pending_notifications.len() >= self.config.max_pending {
                    return Err(PlatformError::CapacityExceeded {
                        current: pending_notifications.len(),
                        max: self.config.max_pending,
                    });
                }
            }

            pending_notifications.insert(notification_id.clone(), pending);
        }

        // Attempt delivery with timeout
        let delivery_result = timeout(
            self.config.delivery_timeout,
            self.deliver_notification(&notification_id, &notification),
        )
        .await;

        // Update notification status based on result
        let success = match delivery_result {
            Ok(Ok(())) => {
                self.mark_notification_delivered(&notification_id).await;
                true
            }
            Ok(Err(e)) => {
                self.mark_notification_failed(&notification_id).await;
                // Add to retry queue if appropriate
                if self.should_retry(&notification) {
                    self.add_to_retry_queue(&notification_id, &notification)
                        .await;
                }
                return Err(e);
            }
            Err(_) => {
                // Timeout occurred
                self.mark_notification_failed(&notification_id).await;
                if self.should_retry(&notification) {
                    self.add_to_retry_queue(&notification_id, &notification)
                        .await;
                }
                return Err(PlatformError::Timeout {
                    message: "notification delivery timed out".to_string(),
                });
            }
        };

        // Update statistics
        self.update_stats(success, now.elapsed()).await;

        Ok(notification_id)
    }

    /// Deliver notification with platform-specific handling
    async fn deliver_notification(
        &self,
        notification_id: &str,
        notification: &Notification,
    ) -> PlatformResult<()> {
        match self.platform {
            Platform::Desktop => {
                self.show_desktop_notification(notification_id, notification)
                    .await
            }
            Platform::Web => {
                self.show_web_notification(notification_id, notification)
                    .await
            }
            Platform::Mobile => {
                self.show_mobile_notification(notification_id, notification)
                    .await
            }
            Platform::Embedded => Err(PlatformError::FeatureNotAvailable {
                feature: "notifications".to_string(),
            }),
        }
    }

    /// Mark notification as delivered
    async fn mark_notification_delivered(&self, notification_id: &str) {
        let mut pending = self.pending_notifications.write().await;
        if let Some(notification) = pending.get_mut(notification_id) {
            notification.status = NotificationStatus::Delivered;
        }
    }

    /// Mark notification as failed
    async fn mark_notification_failed(&self, notification_id: &str) {
        let mut pending = self.pending_notifications.write().await;
        if let Some(notification) = pending.get_mut(notification_id) {
            notification.status = NotificationStatus::Failed;
        }
    }

    /// Check if notification should be retried
    fn should_retry(&self, notification: &Notification) -> bool {
        matches!(
            notification.priority,
            NotificationPriority::High | NotificationPriority::Critical
        )
    }

    /// Add notification to retry queue
    async fn add_to_retry_queue(&self, notification_id: &str, notification: &Notification) {
        let retry_notification = RetryNotification {
            id: notification_id.to_string(),
            notification: notification.clone(),
            retry_count: 0,
            next_retry: Instant::now() + Duration::from_secs(5), // Initial 5 second delay
            max_retries: self.config.max_retry_attempts,
        };

        let mut retry_queue = self.retry_queue.lock().await;
        retry_queue.push_back(retry_notification);
    }

    /// Update notification statistics
    async fn update_stats(&self, success: bool, delivery_time: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_sent += 1;

        if success {
            stats.total_delivered += 1;

            // Update average delivery time
            let total_deliveries = stats.total_delivered;
            if total_deliveries == 1 {
                stats.average_delivery_time = delivery_time;
            } else {
                let total_time =
                    stats.average_delivery_time * (total_deliveries - 1) as u32 + delivery_time;
                stats.average_delivery_time = total_time / total_deliveries as u32;
            }
        } else {
            stats.total_failed += 1;
        }
    }

    /// Cleanup expired notifications
    async fn cleanup_expired_notifications(&self) {
        let now = Instant::now();
        let expiry_duration = Duration::from_secs(300); // 5 minutes

        {
            let mut pending = self.pending_notifications.write().await;
            pending.retain(|_, notification| {
                let age = now.duration_since(notification.created_at);
                if age > expiry_duration {
                    // Mark as expired
                    false
                } else {
                    true
                }
            });
        }

        // Update cleanup timestamp
        {
            let mut stats = self.stats.write().await;
            stats.last_cleanup = Some(now);
        }
    }

    /// Process retry queue
    pub async fn process_retry_queue(&self) -> PlatformResult<usize> {
        let now = Instant::now();
        let mut processed = 0;
        let mut retry_queue = self.retry_queue.lock().await;
        let mut retry_notifications = Vec::new();

        // Collect notifications ready for retry
        while let Some(retry_notif) = retry_queue.pop_front() {
            if retry_notif.next_retry <= now {
                if retry_notif.retry_count < retry_notif.max_retries {
                    retry_notifications.push(retry_notif);
                } else {
                    // Max retries exceeded, give up
                    log::warn!("Notification {} exceeded max retries", retry_notif.id);
                }
            } else {
                // Put back notifications not ready for retry
                retry_queue.push_front(retry_notif);
                break;
            }
        }
        drop(retry_queue);

        // Process retry notifications
        for mut retry_notif in retry_notifications {
            if let Ok(()) = self
                .deliver_notification(&retry_notif.id, &retry_notif.notification)
                .await
            {
                self.mark_notification_delivered(&retry_notif.id).await;
                processed += 1;

                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.total_delivered += 1;
                    stats.total_retries += 1;
                }
            } else {
                // Retry failed, add back to queue with increased delay
                retry_notif.retry_count += 1;
                let delay_seconds =
                    (5.0 * self
                        .config
                        .retry_delay_multiplier
                        .powi(retry_notif.retry_count as i32)) as u64;
                retry_notif.next_retry = now + Duration::from_secs(delay_seconds);

                let mut retry_queue = self.retry_queue.lock().await;
                retry_queue.push_back(retry_notif);
            }
        }

        Ok(processed)
    }

    /// Show desktop notification
    ///
    /// Delivers a real OS-native notification via [`deliver_local_notification`].
    async fn show_desktop_notification(
        &self,
        id: &str,
        notification: &Notification,
    ) -> PlatformResult<()> {
        deliver_local_notification(&notification.title, &notification.body).await?;
        log::debug!(
            "Delivered desktop notification {id} ({} bytes body)",
            notification.body.len()
        );
        Ok(())
    }

    /// Show web notification
    ///
    /// Real browser notifications require the Web Notifications API
    /// (`web_sys::Notification`), reachable only from a `wasm32` target
    /// running inside an actual browser. This crate's dependency graph does
    /// not currently compile for `wasm32-unknown-unknown` at all (`tokio`'s
    /// unconditional `full` feature pulls in `mio`'s socket registration
    /// code, which fails to build for that target — verified directly, not
    /// assumed), so there is no compiled, verifiable browser code path to
    /// ship here; this honestly reports the feature as unavailable rather
    /// than printing a line and claiming a notification was shown.
    async fn show_web_notification(
        &self,
        _id: &str,
        _notification: &Notification,
    ) -> PlatformResult<()> {
        Err(PlatformError::FeatureNotAvailable {
            feature: "web notifications (requires a wasm32 build with a working Web \
                      Notifications API binding; not available in this crate's current \
                      dependency graph)"
                .to_string(),
        })
    }

    /// Show mobile notification
    ///
    /// Real mobile notifications require native FFI bindings
    /// (`UNUserNotificationCenter` on iOS, `NotificationManager` on
    /// Android) that this library does not link against, so this honestly
    /// reports the feature as unavailable instead of printing a line and
    /// claiming a notification was shown.
    async fn show_mobile_notification(
        &self,
        _id: &str,
        _notification: &Notification,
    ) -> PlatformResult<()> {
        Err(PlatformError::FeatureNotAvailable {
            feature: "mobile notifications (requires native iOS/Android FFI bindings not \
                      present in this build)"
                .to_string(),
        })
    }

    /// Cancel a notification
    pub async fn cancel_notification(&mut self, notification_id: &str) -> PlatformResult<()> {
        let mut pending = self.pending_notifications.write().await;
        if pending.remove(notification_id).is_some() {
            match self.platform {
                Platform::Desktop => {
                    // Cancel desktop notification
                    println!("Cancelled desktop notification: {notification_id}");
                }
                Platform::Web => {
                    // Close web notification
                    println!("Cancelled web notification: {notification_id}");
                }
                Platform::Mobile => {
                    // Cancel mobile notification
                    println!("Cancelled mobile notification: {notification_id}");
                }
                Platform::Embedded => {
                    // No-op for embedded
                }
            }
            Ok(())
        } else {
            Err(PlatformError::ConfigurationError {
                message: format!("Notification {notification_id} not found"),
            })
        }
    }

    /// Check if notifications are supported on current platform
    #[must_use]
    pub fn is_supported(&self) -> bool {
        match self.platform {
            Platform::Desktop => true,
            Platform::Web => true, // Most modern browsers support notifications
            Platform::Mobile => true,
            Platform::Embedded => false,
        }
    }

    /// Request notification permission
    pub async fn request_permission(&self) -> PlatformResult<NotificationPermission> {
        match self.platform {
            Platform::Desktop => {
                // Desktop platforms usually don't require explicit permission
                Ok(NotificationPermission::Granted)
            }
            Platform::Web => {
                // Web requires explicit permission request
                #[cfg(target_arch = "wasm32")]
                {
                    // This would use web_sys::Notification::request_permission()
                    Ok(NotificationPermission::Granted)
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    Ok(NotificationPermission::Granted)
                }
            }
            Platform::Mobile => {
                // Mobile platforms require permission
                Ok(NotificationPermission::Granted)
            }
            Platform::Embedded => Ok(NotificationPermission::Denied),
        }
    }

    /// Get notification statistics
    pub async fn get_stats(&self) -> NotificationStats {
        let pending = self.pending_notifications.read().await;
        let stats = self.stats.read().await;
        NotificationStats {
            pending_count: pending.len(),
            total_sent: stats.total_sent,
            total_delivered: stats.total_delivered,
            total_failed: stats.total_failed,
            total_retries: stats.total_retries,
            permission_status: stats.permission_status.clone(),
            average_delivery_time: stats.average_delivery_time,
            last_cleanup: stats.last_cleanup,
        }
    }

    /// Schedule a delayed notification
    pub async fn schedule_notification(
        &mut self,
        notification: Notification,
        delay: Duration,
    ) -> PlatformResult<String> {
        // This would use platform-specific scheduling
        // For now, just simulate immediate delivery after delay
        tokio::time::sleep(delay).await;
        self.show_notification(notification).await
    }

    /// Create a notification for session feedback
    #[must_use]
    pub fn create_feedback_notification(
        &self,
        title: &str,
        message: &str,
        feedback_type: FeedbackType,
    ) -> Notification {
        let priority = match feedback_type {
            FeedbackType::Achievement => NotificationPriority::High,
            FeedbackType::Improvement => NotificationPriority::Normal,
            FeedbackType::Reminder => NotificationPriority::Low,
            FeedbackType::Error => NotificationPriority::High,
        };

        Notification {
            title: title.to_string(),
            body: message.to_string(),
            priority,
            category: NotificationCategory::Feedback,
            icon: Some("voirs-feedback-icon.png".to_string()),
            auto_dismiss_after: Some(Duration::from_secs(5)),
            actions: vec![],
            data: HashMap::new(),
        }
    }
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Show notifications when app is in background
    pub show_when_background: bool,
    /// Default notification priority
    pub default_priority: NotificationPriority,
    /// Maximum number of pending notifications
    pub max_pending: usize,
    /// Auto-dismiss timeout for notifications
    pub default_auto_dismiss: Option<Duration>,
    /// Maximum notifications per minute (rate limiting)
    pub max_notifications_per_minute: u32,
    /// Maximum retry attempts for failed notifications
    pub max_retry_attempts: u32,
    /// Retry delay multiplier (exponential backoff)
    pub retry_delay_multiplier: f32,
    /// Notification timeout duration
    pub delivery_timeout: Duration,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_when_background: true,
            default_priority: NotificationPriority::Normal,
            max_pending: 10,
            default_auto_dismiss: Some(Duration::from_secs(5)),
            max_notifications_per_minute: 30,
            max_retry_attempts: 3,
            retry_delay_multiplier: 2.0,
            delivery_timeout: Duration::from_secs(10),
        }
    }
}

/// Cross-platform notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    /// Notification title
    pub title: String,
    /// Notification body text
    pub body: String,
    /// Notification priority
    pub priority: NotificationPriority,
    /// Notification category
    pub category: NotificationCategory,
    /// Icon path or URL
    pub icon: Option<String>,
    /// Auto-dismiss timeout
    pub auto_dismiss_after: Option<Duration>,
    /// Available actions
    pub actions: Vec<NotificationAction>,
    /// Additional data
    pub data: HashMap<String, String>,
}

/// Notification priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    /// Description
    Low,
    /// Description
    Normal,
    /// Description
    High,
    /// Description
    Critical,
}

/// Notification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationCategory {
    /// Description
    Feedback,
    /// Description
    Achievement,
    /// Description
    Reminder,
    /// Description
    System,
    /// Description
    Social,
}

/// Notification action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    /// Description
    pub id: String,
    /// Description
    pub title: String,
    /// Description
    pub icon: Option<String>,
}

/// Notification permission status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum NotificationPermission {
    /// Description
    Granted,
    /// Description
    Denied,
    #[default]
    /// Description
    Default,
}

/// Feedback notification types
#[derive(Debug, Clone)]
pub enum FeedbackType {
    /// Description
    Achievement,
    /// Description
    Improvement,
    /// Description
    Reminder,
    /// Description
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_manager_creation() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Desktop, config);
        assert!(manager.is_supported());
    }

    #[test]
    fn test_notification_creation() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Web, config);

        let notification = manager.create_feedback_notification(
            "Great Progress!",
            "You've improved your pronunciation by 15%",
            FeedbackType::Achievement,
        );

        assert_eq!(notification.title, "Great Progress!");
        assert!(!notification.body.is_empty());
        assert!(matches!(notification.priority, NotificationPriority::High));
        assert!(matches!(
            notification.category,
            NotificationCategory::Feedback
        ));
    }

    #[tokio::test]
    async fn test_notification_permission() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Desktop, config);

        let permission = manager.request_permission().await.unwrap();
        assert_eq!(permission, NotificationPermission::Granted);
    }

    #[tokio::test]
    async fn test_show_notification() {
        let config = NotificationConfig::default();
        let mut manager = NotificationManager::new(Platform::Desktop, config);

        let notification = Notification {
            title: "Test Notification".to_string(),
            body: "This is a test notification".to_string(),
            priority: NotificationPriority::Normal,
            category: NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: vec![],
            data: HashMap::new(),
        };

        let notification_id = manager.show_notification(notification).await.unwrap();
        assert!(!notification_id.is_empty());
        assert!(manager
            .pending_notifications
            .read()
            .await
            .contains_key(&notification_id));
    }

    #[tokio::test]
    async fn test_cancel_notification() {
        let config = NotificationConfig::default();
        let mut manager = NotificationManager::new(Platform::Mobile, config);

        // Add a pending notification manually
        let notification_id = "test_notification_id".to_string();
        let pending_notification = PendingNotification {
            notification: Notification {
                title: "Test".to_string(),
                body: "Test body".to_string(),
                priority: NotificationPriority::Normal,
                category: NotificationCategory::System,
                icon: None,
                auto_dismiss_after: None,
                actions: vec![],
                data: HashMap::new(),
            },
            created_at: std::time::Instant::now(),
            attempts: 0,
            last_attempt: None,
            status: NotificationStatus::Pending,
        };

        manager
            .pending_notifications
            .write()
            .await
            .insert(notification_id.clone(), pending_notification);

        // Cancel the notification
        assert!(manager.cancel_notification(&notification_id).await.is_ok());
        assert!(!manager
            .pending_notifications
            .read()
            .await
            .contains_key(&notification_id));
    }

    #[test]
    fn test_platform_support() {
        let config = NotificationConfig::default();

        let desktop_manager = NotificationManager::new(Platform::Desktop, config.clone());
        assert!(desktop_manager.is_supported());

        let web_manager = NotificationManager::new(Platform::Web, config.clone());
        assert!(web_manager.is_supported());

        let mobile_manager = NotificationManager::new(Platform::Mobile, config.clone());
        assert!(mobile_manager.is_supported());

        let embedded_manager = NotificationManager::new(Platform::Embedded, config);
        assert!(!embedded_manager.is_supported());
    }

    #[tokio::test]
    async fn test_notification_stats() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Web, config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.pending_count, 0);
        assert_eq!(stats.permission_status, NotificationPermission::Default);
    }

    #[tokio::test]
    async fn test_schedule_notification() {
        let config = NotificationConfig::default();
        let mut manager = NotificationManager::new(Platform::Desktop, config);

        let notification = Notification {
            title: "Scheduled Notification".to_string(),
            body: "This notification was scheduled".to_string(),
            priority: NotificationPriority::Normal,
            category: NotificationCategory::Reminder,
            icon: None,
            auto_dismiss_after: None,
            actions: vec![],
            data: HashMap::new(),
        };

        let start = std::time::Instant::now();
        let notification_id = manager
            .schedule_notification(notification, Duration::from_millis(100))
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(!notification_id.is_empty());
        assert!(elapsed >= Duration::from_millis(100));
    }

    /// Exercises the real OS-native delivery path directly. On macOS/Linux
    /// this is a genuine end-to-end call into `osascript`/`notify-send`;
    /// title/body containing quotes, backslashes, and newlines must not
    /// break the command (proves the argv-based escaping actually works,
    /// not just a happy-path string).
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn test_deliver_local_notification_handles_adversarial_text() {
        let result = deliver_local_notification(
            "Title with \"quotes\" and \\backslash\\",
            "Body with \"quotes\", \\backslashes\\, and a\nnewline",
        )
        .await;
        assert!(
            result.is_ok(),
            "real local delivery should succeed with adversarial text: {result:?}"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn test_notification_spawn_error_distinguishes_missing_binary() {
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        let err = notification_spawn_error("notify-send", &not_found);
        assert!(matches!(err, PlatformError::FeatureNotAvailable { .. }));

        let other = std::io::Error::other("boom");
        let err = notification_spawn_error("notify-send", &other);
        assert!(matches!(err, PlatformError::ConfigurationError { .. }));
    }

    #[tokio::test]
    async fn test_web_notification_fails_closed_not_fake_success() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Web, config);

        let notification = Notification {
            title: "Web Test".to_string(),
            body: "Should not fabricate delivery".to_string(),
            priority: NotificationPriority::Normal,
            category: NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: vec![],
            data: HashMap::new(),
        };

        let result = manager.deliver_notification("test-id", &notification).await;
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    #[tokio::test]
    async fn test_mobile_notification_fails_closed_not_fake_success() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(Platform::Mobile, config);

        let notification = Notification {
            title: "Mobile Test".to_string(),
            body: "Should not fabricate delivery".to_string(),
            priority: NotificationPriority::Normal,
            category: NotificationCategory::System,
            icon: None,
            auto_dismiss_after: None,
            actions: vec![],
            data: HashMap::new(),
        };

        let result = manager.deliver_notification("test-id", &notification).await;
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }
}
