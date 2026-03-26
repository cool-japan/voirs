// src/usage_analytics.rs
//! Comprehensive Usage Analytics for `VoiRS` Feedback System
//!
//! This module provides detailed analytics and tracking capabilities to monitor
//! system usage, user behavior, feature adoption, and performance metrics. It enables
//! data-driven decision making and system optimization.
//!
//! # Features
//! - API endpoint usage tracking
//! - Feature adoption metrics
//! - User engagement patterns
//! - Session analytics
//! - Time-series data analysis
//! - Cohort analysis
//! - Funnel analysis
//! - Real-time and historical reporting
//!
//! # Examples
//! ```
//! use voirs_feedback::usage_analytics::{UsageAnalytics, AnalyticsEvent, EventType};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create analytics tracker
//! let mut analytics = UsageAnalytics::new();
//!
//! // Track an event
//! analytics.track_event(AnalyticsEvent {
//!     event_type: EventType::ApiCall,
//!     user_id: Some("user123".to_string()),
//!     endpoint: Some("/api/feedback".to_string()),
//!     metadata: serde_json::json!({"method": "POST", "status": 200}),
//! }).await?;
//!
//! // Get usage report
//! let report = analytics.get_usage_report(std::time::Duration::from_secs(3600)).await?;
//! println!("API calls in last hour: {}", report.total_api_calls);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Datelike, Duration as ChronoDuration, Timelike, Utc};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

/// Analytics errors
#[derive(Debug, Error)]
pub enum AnalyticsError {
    /// Invalid time range
    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),

    /// Data not found
    #[error("Analytics data not found for: {0}")]
    DataNotFound(String),

    /// Calculation error
    #[error("Analytics calculation error: {0}")]
    CalculationError(String),
}

/// Result type for analytics operations
pub type AnalyticsResult<T> = Result<T, AnalyticsError>;

/// Event types for analytics tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// API endpoint called
    ApiCall,
    /// User session started
    SessionStarted,
    /// User session ended
    SessionEnded,
    /// Feedback generated
    FeedbackGenerated,
    /// Training exercise completed
    ExerciseCompleted,
    /// Achievement unlocked
    AchievementUnlocked,
    /// Feature used
    FeatureUsed,
    /// Error occurred
    ErrorOccurred,
    /// Page viewed
    PageView,
    /// Button clicked
    ButtonClick,
    /// Custom event
    Custom(String),
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiCall => write!(f, "api_call"),
            Self::SessionStarted => write!(f, "session_started"),
            Self::SessionEnded => write!(f, "session_ended"),
            Self::FeedbackGenerated => write!(f, "feedback_generated"),
            Self::ExerciseCompleted => write!(f, "exercise_completed"),
            Self::AchievementUnlocked => write!(f, "achievement_unlocked"),
            Self::FeatureUsed => write!(f, "feature_used"),
            Self::ErrorOccurred => write!(f, "error_occurred"),
            Self::PageView => write!(f, "page_view"),
            Self::ButtonClick => write!(f, "button_click"),
            Self::Custom(name) => write!(f, "custom.{name}"),
        }
    }
}

/// Analytics event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    /// Type of event
    pub event_type: EventType,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Endpoint or feature name
    pub endpoint: Option<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
}

/// Usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReport {
    /// Time period start
    pub period_start: DateTime<Utc>,
    /// Time period end
    pub period_end: DateTime<Utc>,
    /// Total API calls
    pub total_api_calls: u64,
    /// Unique users
    pub unique_users: u64,
    /// Total sessions
    pub total_sessions: u64,
    /// Average session duration (seconds)
    pub avg_session_duration: f64,
    /// Most popular endpoints
    pub top_endpoints: Vec<(String, u64)>,
    /// Most active users
    pub top_users: Vec<(String, u64)>,
    /// Hourly distribution
    pub hourly_distribution: Vec<(u8, u64)>,
    /// Daily distribution
    pub daily_distribution: Vec<(String, u64)>,
}

/// Feature usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetrics {
    /// Feature name
    pub feature_name: String,
    /// Total usage count
    pub usage_count: u64,
    /// Unique users
    pub unique_users: u64,
    /// Adoption rate (percentage)
    pub adoption_rate: f64,
    /// Average uses per user
    pub avg_uses_per_user: f64,
    /// First seen
    pub first_seen: DateTime<Utc>,
    /// Last seen
    pub last_seen: DateTime<Utc>,
}

/// Cohort analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortAnalysis {
    /// Cohort name (e.g., "2024-01")
    pub cohort_name: String,
    /// Cohort start date
    pub cohort_start: DateTime<Utc>,
    /// Total users in cohort
    pub total_users: u64,
    /// Retention by period
    pub retention: HashMap<u32, f64>,
}

/// Funnel analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelAnalysis {
    /// Funnel stages
    pub stages: Vec<FunnelStage>,
    /// Overall conversion rate
    pub overall_conversion: f64,
}

/// Funnel stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelStage {
    /// Stage name
    pub name: String,
    /// Users who reached this stage
    pub users: u64,
    /// Conversion rate from previous stage
    pub conversion_rate: f64,
    /// Drop-off from previous stage
    pub drop_off: u64,
}

/// Internal event record with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventRecord {
    event: AnalyticsEvent,
    timestamp: DateTime<Utc>,
}

/// Usage analytics tracker
pub struct UsageAnalytics {
    events: Arc<RwLock<Vec<EventRecord>>>,
    session_start_times: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    feature_usage: Arc<RwLock<HashMap<String, Vec<(DateTime<Utc>, String)>>>>,
    max_events: usize,
}

impl UsageAnalytics {
    /// Create a new usage analytics tracker
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(1000000) // 1 million events by default
    }

    /// Create with specific capacity
    #[must_use]
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::with_capacity(max_events / 10))),
            session_start_times: Arc::new(RwLock::new(HashMap::new())),
            feature_usage: Arc::new(RwLock::new(HashMap::new())),
            max_events,
        }
    }

    /// Track an analytics event
    pub async fn track_event(&mut self, event: AnalyticsEvent) -> AnalyticsResult<()> {
        let timestamp = Utc::now();

        // Handle session start/end
        match event.event_type {
            EventType::SessionStarted => {
                if let Some(user_id) = &event.user_id {
                    let mut session_times = self.session_start_times.write().await;
                    session_times.insert(user_id.clone(), timestamp);
                }
            }
            EventType::SessionEnded => {
                if let Some(user_id) = &event.user_id {
                    let mut session_times = self.session_start_times.write().await;
                    session_times.remove(user_id);
                }
            }
            EventType::FeatureUsed => {
                if let (Some(user_id), Some(feature)) = (&event.user_id, &event.endpoint) {
                    let mut feature_usage = self.feature_usage.write().await;
                    feature_usage
                        .entry(feature.clone())
                        .or_insert_with(Vec::new)
                        .push((timestamp, user_id.clone()));
                }
            }
            _ => {}
        }

        // Store event
        let record = EventRecord { event, timestamp };

        let mut events = self.events.write().await;
        events.push(record);

        // Cleanup old events if needed
        if events.len() > self.max_events {
            let remove_count = events.len() - self.max_events;
            events.drain(0..remove_count);
        }

        Ok(())
    }

    /// Get usage report for a time period
    pub async fn get_usage_report(&self, period: Duration) -> AnalyticsResult<UsageReport> {
        let now = Utc::now();
        let period_start = now
            - ChronoDuration::from_std(period)
                .map_err(|e| AnalyticsError::InvalidTimeRange(format!("Invalid duration: {e}")))?;

        let events = self.events.read().await;

        // Filter events in period
        let period_events: Vec<_> = events
            .iter()
            .filter(|e| e.timestamp >= period_start && e.timestamp <= now)
            .collect();

        // Calculate API calls
        let total_api_calls = period_events
            .iter()
            .filter(|e| matches!(e.event.event_type, EventType::ApiCall))
            .count() as u64;

        // Calculate unique users
        let unique_users = period_events
            .iter()
            .filter_map(|e| e.event.user_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;

        // Calculate total sessions
        let total_sessions = period_events
            .iter()
            .filter(|e| matches!(e.event.event_type, EventType::SessionStarted))
            .count() as u64;

        // Calculate average session duration
        let session_durations: Vec<f64> = period_events
            .iter()
            .filter(|e| matches!(e.event.event_type, EventType::SessionStarted))
            .filter_map(|start| {
                let user_id = start.event.user_id.as_ref()?;
                // Find corresponding end event
                period_events
                    .iter()
                    .find(|e| {
                        matches!(e.event.event_type, EventType::SessionEnded)
                            && e.event.user_id.as_ref() == Some(user_id)
                            && e.timestamp > start.timestamp
                    })
                    .map(|end| (end.timestamp - start.timestamp).num_seconds() as f64)
            })
            .collect();

        let avg_session_duration = if session_durations.is_empty() {
            0.0
        } else {
            session_durations.iter().sum::<f64>() / session_durations.len() as f64
        };

        // Calculate top endpoints
        let mut endpoint_counts: HashMap<String, u64> = HashMap::new();
        for event in &period_events {
            if let Some(endpoint) = &event.event.endpoint {
                *endpoint_counts.entry(endpoint.clone()).or_insert(0) += 1;
            }
        }

        let mut top_endpoints: Vec<_> = endpoint_counts.into_iter().collect();
        top_endpoints.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_endpoints.truncate(10);

        // Calculate top users
        let mut user_counts: HashMap<String, u64> = HashMap::new();
        for event in &period_events {
            if let Some(user_id) = &event.event.user_id {
                *user_counts.entry(user_id.clone()).or_insert(0) += 1;
            }
        }

        let mut top_users: Vec<_> = user_counts.into_iter().collect();
        top_users.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_users.truncate(10);

        // Calculate hourly distribution
        let mut hourly_counts: HashMap<u8, u64> = HashMap::new();
        for event in &period_events {
            let hour = event.timestamp.hour() as u8;
            *hourly_counts.entry(hour).or_insert(0) += 1;
        }

        let hourly_distribution: Vec<_> = (0..24)
            .map(|hour| (hour, *hourly_counts.get(&hour).unwrap_or(&0)))
            .collect();

        // Calculate daily distribution
        let mut daily_counts: HashMap<String, u64> = HashMap::new();
        for event in &period_events {
            let day = event.timestamp.format("%Y-%m-%d").to_string();
            *daily_counts.entry(day).or_insert(0) += 1;
        }

        let mut daily_distribution: Vec<_> = daily_counts.into_iter().collect();
        daily_distribution.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(UsageReport {
            period_start,
            period_end: now,
            total_api_calls,
            unique_users,
            total_sessions,
            avg_session_duration,
            top_endpoints,
            top_users,
            hourly_distribution,
            daily_distribution,
        })
    }

    /// Get feature usage metrics
    pub async fn get_feature_metrics(&self, feature_name: &str) -> AnalyticsResult<FeatureMetrics> {
        let feature_usage = self.feature_usage.read().await;

        let usage_data = feature_usage
            .get(feature_name)
            .ok_or_else(|| AnalyticsError::DataNotFound(feature_name.to_string()))?;

        let usage_count = usage_data.len() as u64;
        let unique_users = usage_data
            .iter()
            .map(|(_, user)| user)
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;

        let first_seen = usage_data.first().map_or(Utc::now(), |(ts, _)| *ts);
        let last_seen = usage_data.last().map_or(Utc::now(), |(ts, _)| *ts);

        let avg_uses_per_user = if unique_users > 0 {
            usage_count as f64 / unique_users as f64
        } else {
            0.0
        };

        // Calculate adoption rate (percentage of active users using this feature)
        let events = self.events.read().await;
        let total_users = events
            .iter()
            .filter_map(|e| e.event.user_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;

        let adoption_rate = if total_users > 0 {
            (unique_users as f64 / total_users as f64) * 100.0
        } else {
            0.0
        };

        Ok(FeatureMetrics {
            feature_name: feature_name.to_string(),
            usage_count,
            unique_users,
            adoption_rate,
            avg_uses_per_user,
            first_seen,
            last_seen,
        })
    }

    /// Perform cohort analysis
    pub async fn cohort_analysis(
        &self,
        cohort_period: ChronoDuration,
    ) -> AnalyticsResult<Vec<CohortAnalysis>> {
        let events = self.events.read().await;

        // Group users by cohort (first seen date)
        let mut user_cohorts: HashMap<String, DateTime<Utc>> = HashMap::new();

        for event in events.iter() {
            if let Some(user_id) = &event.event.user_id {
                user_cohorts
                    .entry(user_id.clone())
                    .or_insert(event.timestamp);
            }
        }

        // Group cohorts by period
        let mut cohort_groups: HashMap<String, Vec<String>> = HashMap::new();

        for (user_id, first_seen) in &user_cohorts {
            let cohort_name = first_seen.format("%Y-%m").to_string();
            cohort_groups
                .entry(cohort_name)
                .or_default()
                .push(user_id.clone());
        }

        // Calculate retention for each cohort
        let mut cohorts = Vec::new();

        for (cohort_name, users) in &cohort_groups {
            let cohort_start = users
                .iter()
                .filter_map(|u| user_cohorts.get(u))
                .min()
                .copied()
                .unwrap_or(Utc::now());

            let total_users = users.len() as u64;

            // Calculate retention for each period
            let mut retention = HashMap::new();
            for period in 0..12 {
                let period_start = cohort_start + (cohort_period * period as i32);
                let period_end = period_start + cohort_period;

                let active_users = users
                    .iter()
                    .filter(|user_id| {
                        events.iter().any(|e| {
                            e.event.user_id.as_ref() == Some(user_id)
                                && e.timestamp >= period_start
                                && e.timestamp < period_end
                        })
                    })
                    .count() as u64;

                let retention_rate = if total_users > 0 {
                    (active_users as f64 / total_users as f64) * 100.0
                } else {
                    0.0
                };

                retention.insert(period, retention_rate);
            }

            cohorts.push(CohortAnalysis {
                cohort_name: cohort_name.clone(),
                cohort_start,
                total_users,
                retention,
            });
        }

        cohorts.sort_by_key(|a| a.cohort_start);

        Ok(cohorts)
    }

    /// Perform funnel analysis
    pub async fn funnel_analysis(
        &self,
        stage_events: Vec<EventType>,
    ) -> AnalyticsResult<FunnelAnalysis> {
        if stage_events.is_empty() {
            return Err(AnalyticsError::InvalidTimeRange(
                "No stages provided".to_string(),
            ));
        }

        let events = self.events.read().await;

        let mut stages = Vec::new();
        let mut previous_users: Option<std::collections::HashSet<String>> = None;
        let total_users_at_start = events
            .iter()
            .filter(|e| e.event.event_type == stage_events[0])
            .filter_map(|e| e.event.user_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;

        for (idx, event_type) in stage_events.iter().enumerate() {
            // Get users who completed this stage
            let stage_users: std::collections::HashSet<String> = events
                .iter()
                .filter(|e| e.event.event_type == *event_type)
                .filter_map(|e| e.event.user_id.clone())
                .collect();

            let users = stage_users.len() as u64;

            let (conversion_rate, drop_off) = if let Some(prev_users) = &previous_users {
                let conversion = if prev_users.is_empty() {
                    0.0
                } else {
                    (users as f64 / prev_users.len() as f64) * 100.0
                };
                let drop = prev_users.len() as u64 - users;
                (conversion, drop)
            } else {
                (100.0, 0)
            };

            stages.push(FunnelStage {
                name: event_type.to_string(),
                users,
                conversion_rate,
                drop_off,
            });

            previous_users = Some(stage_users);
        }

        let overall_conversion = if total_users_at_start > 0 && !stages.is_empty() {
            (stages.last().expect("collection should not be empty").users as f64
                / total_users_at_start as f64)
                * 100.0
        } else {
            0.0
        };

        Ok(FunnelAnalysis {
            stages,
            overall_conversion,
        })
    }

    /// Get time series data for an event type
    pub async fn get_time_series(
        &self,
        event_type: EventType,
        interval: Duration,
        period: Duration,
    ) -> AnalyticsResult<Vec<TimeSeriesPoint>> {
        let now = Utc::now();
        let period_start = now
            - ChronoDuration::from_std(period)
                .map_err(|e| AnalyticsError::InvalidTimeRange(format!("Invalid period: {e}")))?;

        let interval_chrono = ChronoDuration::from_std(interval)
            .map_err(|e| AnalyticsError::InvalidTimeRange(format!("Invalid interval: {e}")))?;

        let events = self.events.read().await;

        // Create time buckets
        let mut buckets: HashMap<DateTime<Utc>, u64> = HashMap::new();

        for event in events.iter() {
            if event.timestamp >= period_start
                && event.timestamp <= now
                && event.event.event_type == event_type
            {
                // Round timestamp to interval
                let bucket = period_start
                    + interval_chrono
                        * ((event.timestamp - period_start).num_seconds()
                            / interval_chrono.num_seconds()) as i32;

                *buckets.entry(bucket).or_insert(0) += 1;
            }
        }

        // Create time series with all intervals
        let mut time_series = Vec::new();
        let mut current = period_start;

        while current <= now {
            time_series.push(TimeSeriesPoint {
                timestamp: current,
                value: *buckets.get(&current).unwrap_or(&0) as f64,
            });
            current += interval_chrono;
        }

        Ok(time_series)
    }

    /// Clear all analytics data
    pub async fn clear(&mut self) {
        let mut events = self.events.write().await;
        events.clear();

        let mut session_times = self.session_start_times.write().await;
        session_times.clear();

        let mut feature_usage = self.feature_usage.write().await;
        feature_usage.clear();
    }
}

impl Default for UsageAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_track_event() {
        let mut analytics = UsageAnalytics::new();

        let event = AnalyticsEvent {
            event_type: EventType::ApiCall,
            user_id: Some("user1".to_string()),
            endpoint: Some("/api/feedback".to_string()),
            metadata: serde_json::json!({"status": 200}),
        };

        let result = analytics.track_event(event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_usage_report() {
        let mut analytics = UsageAnalytics::new();

        // Track some events
        for i in 0..10 {
            analytics
                .track_event(AnalyticsEvent {
                    event_type: EventType::ApiCall,
                    user_id: Some(format!("user{}", i % 3)),
                    endpoint: Some("/api/feedback".to_string()),
                    metadata: serde_json::json!({}),
                })
                .await
                .unwrap();
        }

        let report = analytics
            .get_usage_report(Duration::from_secs(3600))
            .await
            .unwrap();
        assert_eq!(report.total_api_calls, 10);
        assert_eq!(report.unique_users, 3);
    }

    #[tokio::test]
    async fn test_feature_metrics() {
        let mut analytics = UsageAnalytics::new();

        // Track feature usage
        for i in 0..5 {
            analytics
                .track_event(AnalyticsEvent {
                    event_type: EventType::FeatureUsed,
                    user_id: Some(format!("user{}", i % 2)),
                    endpoint: Some("gamification".to_string()),
                    metadata: serde_json::json!({}),
                })
                .await
                .unwrap();
        }

        let metrics = analytics.get_feature_metrics("gamification").await.unwrap();
        assert_eq!(metrics.usage_count, 5);
        assert_eq!(metrics.unique_users, 2);
    }

    #[tokio::test]
    async fn test_time_series() {
        let mut analytics = UsageAnalytics::new();

        // Track events
        for _ in 0..10 {
            analytics
                .track_event(AnalyticsEvent {
                    event_type: EventType::FeedbackGenerated,
                    user_id: Some("user1".to_string()),
                    endpoint: None,
                    metadata: serde_json::json!({}),
                })
                .await
                .unwrap();
        }

        let time_series = analytics
            .get_time_series(
                EventType::FeedbackGenerated,
                Duration::from_secs(60),
                Duration::from_secs(3600),
            )
            .await
            .unwrap();

        assert!(!time_series.is_empty());
    }

    #[tokio::test]
    async fn test_event_type_display() {
        assert_eq!(EventType::ApiCall.to_string(), "api_call");
        assert_eq!(EventType::SessionStarted.to_string(), "session_started");
        assert_eq!(EventType::FeatureUsed.to_string(), "feature_used");
        assert_eq!(
            EventType::Custom("test".to_string()).to_string(),
            "custom.test"
        );
    }

    #[tokio::test]
    async fn test_clear() {
        let mut analytics = UsageAnalytics::new();

        analytics
            .track_event(AnalyticsEvent {
                event_type: EventType::ApiCall,
                user_id: Some("user1".to_string()),
                endpoint: Some("/api/feedback".to_string()),
                metadata: serde_json::json!({}),
            })
            .await
            .unwrap();

        analytics.clear().await;

        let report = analytics
            .get_usage_report(Duration::from_secs(3600))
            .await
            .unwrap();
        assert_eq!(report.total_api_calls, 0);
    }
}
