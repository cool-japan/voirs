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

/// Pending notification with tracking information
#[derive(Debug, Clone)]
pub struct PendingNotification {
    pub notification: Notification,
    pub created_at: Instant,
    pub attempts: u32,
    pub last_attempt: Option<Instant>,
    pub status: NotificationStatus,
}

/// Retry notification for failed delivery attempts
#[derive(Debug, Clone)]
pub struct RetryNotification {
    pub id: String,
    pub notification: Notification,
    pub retry_count: u32,
    pub next_retry: Instant,
    pub max_retries: u32,
}

/// Notification delivery status
#[derive(Debug, Clone, PartialEq)]
pub enum NotificationStatus {
    Pending,
    Delivering,
    Delivered,
    Failed,
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
    pub fn new(max_notifications_per_minute: u32) -> Self {
        Self {
            tokens: max_notifications_per_minute,
            last_refill: Instant::now(),
            max_tokens: max_notifications_per_minute,
            refill_rate: Duration::from_secs(60),
        }
    }

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
    pub pending_count: usize,
    pub total_sent: u64,
    pub total_delivered: u64,
    pub total_failed: u64,
    pub total_retries: u64,
    pub permission_status: NotificationPermission,
    pub average_delivery_time: Duration,
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
            Ok(Ok(_)) => {
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
            match self
                .deliver_notification(&retry_notif.id, &retry_notif.notification)
                .await
            {
                Ok(_) => {
                    self.mark_notification_delivered(&retry_notif.id).await;
                    processed += 1;

                    // Update stats
                    {
                        let mut stats = self.stats.write().await;
                        stats.total_delivered += 1;
                        stats.total_retries += 1;
                    }
                }
                Err(_) => {
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
        }

        Ok(processed)
    }

    /// Show desktop notification
    async fn show_desktop_notification(
        &self,
        id: &str,
        notification: &Notification,
    ) -> PlatformResult<()> {
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        {
            // On desktop platforms, we would use platform-specific notification APIs
            // Windows: Windows Runtime API (WinRT)
            // macOS: NSUserNotification or UNUserNotificationCenter
            // Linux: D-Bus notification service

            println!(
                "Desktop Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );

            // Simulate platform-specific notification display
            if let Some(icon_path) = &notification.icon {
                println!("  Icon: {}", icon_path);
            }

            if let Some(duration) = notification.auto_dismiss_after {
                println!("  Auto-dismiss after: {:?}", duration);
            }
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            // Fallback for non-desktop platforms
            println!(
                "Desktop Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        Ok(())
    }

    /// Show web notification
    async fn show_web_notification(
        &self,
        id: &str,
        notification: &Notification,
    ) -> PlatformResult<()> {
        #[cfg(target_arch = "wasm32")]
        {
            // In WASM environment, this would use the Web Notifications API
            // First check if notifications are supported
            // Then request permission if needed
            // Finally create and show the notification

            // let permission = web_sys::Notification::permission();
            // if permission == web_sys::NotificationPermission::Default {
            //     let permission_result = web_sys::Notification::request_permission()
            //         .await
            //         .map_err(|e| PlatformError::FeatureNotAvailable {
            //             feature: format!("notification permission: {:?}", e),
            //         })?;
            // }
            //
            // if permission == web_sys::NotificationPermission::Granted {
            //     let notification_options = web_sys::NotificationOptions::new();
            //     notification_options.set_body(&notification.body);
            //     if let Some(icon) = &notification.icon {
            //         notification_options.set_icon(icon);
            //     }
            //
            //     let web_notification = web_sys::Notification::new_with_options(
            //         &notification.title,
            //         &notification_options
            //     ).map_err(|e| PlatformError::NetworkError {
            //         message: format!("Failed to create notification: {:?}", e),
            //     })?;
            // }

            println!(
                "Web Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            println!(
                "Web Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        Ok(())
    }

    /// Show mobile notification
    async fn show_mobile_notification(
        &self,
        id: &str,
        notification: &Notification,
    ) -> PlatformResult<()> {
        #[cfg(target_os = "ios")]
        {
            // iOS notifications would use UNUserNotificationCenter
            // Request authorization first, then create notification content
            println!(
                "iOS Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        #[cfg(target_os = "android")]
        {
            // Android notifications would use NotificationManager and NotificationChannel
            println!(
                "Android Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            println!(
                "Mobile Notification [{}]: {} - {}",
                id, notification.title, notification.body
            );
        }

        Ok(())
    }

    /// Cancel a notification
    pub async fn cancel_notification(&mut self, notification_id: &str) -> PlatformResult<()> {
        let mut pending = self.pending_notifications.write().await;
        if pending.remove(notification_id).is_some() {
            match self.platform {
                Platform::Desktop => {
                    // Cancel desktop notification
                    println!("Cancelled desktop notification: {}", notification_id);
                }
                Platform::Web => {
                    // Close web notification
                    println!("Cancelled web notification: {}", notification_id);
                }
                Platform::Mobile => {
                    // Cancel mobile notification
                    println!("Cancelled mobile notification: {}", notification_id);
                }
                Platform::Embedded => {
                    // No-op for embedded
                }
            }
            Ok(())
        } else {
            Err(PlatformError::ConfigurationError {
                message: format!("Notification {} not found", notification_id),
            })
        }
    }

    /// Check if notifications are supported on current platform
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
    Low,
    Normal,
    High,
    Critical,
}

/// Notification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationCategory {
    Feedback,
    Achievement,
    Reminder,
    System,
    Social,
}

/// Notification action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    pub id: String,
    pub title: String,
    pub icon: Option<String>,
}

/// Notification permission status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum NotificationPermission {
    Granted,
    Denied,
    #[default]
    Default,
}

/// Feedback notification types
#[derive(Debug, Clone)]
pub enum FeedbackType {
    Achievement,
    Improvement,
    Reminder,
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
}
