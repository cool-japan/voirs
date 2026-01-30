//! Timezone Support Module
//!
//! Provides comprehensive timezone handling for global applications including
//! timezone conversion, DST handling, scheduling across timezones, and
//! localized time display.

use chrono::{
    DateTime, Datelike, Duration, FixedOffset, Local, NaiveDateTime, Offset, TimeZone, Timelike,
    Utc,
};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use thiserror::Error;

/// Timezone errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum TimezoneError {
    /// Invalid timezone
    #[error("Invalid timezone: {timezone}")]
    InvalidTimezone { timezone: String },

    /// Parsing error
    #[error("Failed to parse datetime: {message}")]
    ParsingError { message: String },

    /// Conversion error
    #[error("Timezone conversion failed: {message}")]
    ConversionError { message: String },
}

/// Result type for timezone operations
pub type TimezoneResult<T> = Result<T, TimezoneError>;

/// Timezone information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimezoneInfo {
    /// IANA timezone identifier (e.g., "`America/New_York`")
    pub identifier: String,
    /// Display name
    pub display_name: String,
    /// UTC offset in seconds
    pub utc_offset_seconds: i32,
    /// Whether DST is currently active
    pub is_dst: bool,
    /// DST offset in seconds (if applicable)
    pub dst_offset_seconds: i32,
    /// Abbreviation (e.g., "EST", "EDT")
    pub abbreviation: String,
}

/// Scheduled event with timezone awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// Event ID
    pub event_id: String,
    /// Event title
    pub title: String,
    /// Start time in UTC
    pub start_time_utc: DateTime<Utc>,
    /// End time in UTC
    pub end_time_utc: DateTime<Utc>,
    /// Original timezone
    pub original_timezone: String,
    /// Recurrence rule (RFC 5545 format)
    pub recurrence: Option<String>,
}

/// Time display format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum TimeFormat {
    /// 12-hour format (e.g., "2:30 PM")
    Hour12,
    /// 24-hour format (e.g., "14:30")
    Hour24,
    /// ISO 8601 (e.g., "2023-12-04T14:30:00Z")
    Iso8601,
    /// Custom format string
    Custom { format: String },
}

/// Timezone manager
pub struct TimezoneManager {
    /// User timezone preferences (`user_id` -> timezone)
    user_timezones: HashMap<String, String>,
    /// Cached timezone info
    timezone_cache: HashMap<String, TimezoneInfo>,
}

impl TimezoneManager {
    /// Create new timezone manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            user_timezones: HashMap::new(),
            timezone_cache: HashMap::new(),
        }
    }

    /// Convert time from one timezone to another
    pub fn convert_timezone(
        &self,
        datetime: DateTime<Utc>,
        target_timezone: &str,
    ) -> TimezoneResult<DateTime<Tz>> {
        let tz = Tz::from_str(target_timezone).map_err(|_| TimezoneError::InvalidTimezone {
            timezone: target_timezone.to_string(),
        })?;

        Ok(datetime.with_timezone(&tz))
    }

    /// Get timezone information
    pub fn get_timezone_info(&mut self, timezone: &str) -> TimezoneResult<TimezoneInfo> {
        // Check cache
        if let Some(cached) = self.timezone_cache.get(timezone) {
            return Ok(cached.clone());
        }

        let tz = Tz::from_str(timezone).map_err(|_| TimezoneError::InvalidTimezone {
            timezone: timezone.to_string(),
        })?;

        let now = Utc::now();
        let now_in_tz = now.with_timezone(&tz);

        let info = TimezoneInfo {
            identifier: timezone.to_string(),
            display_name: timezone.to_string(),
            utc_offset_seconds: now_in_tz.offset().fix().local_minus_utc(),
            is_dst: self.is_dst(&now_in_tz),
            dst_offset_seconds: 0, // Simplified
            abbreviation: format!("{}", now_in_tz.format("%Z")),
        };

        self.timezone_cache
            .insert(timezone.to_string(), info.clone());
        Ok(info)
    }

    /// Set user timezone preference
    pub fn set_user_timezone(&mut self, user_id: String, timezone: String) -> TimezoneResult<()> {
        // Validate timezone
        Tz::from_str(&timezone).map_err(|_| TimezoneError::InvalidTimezone {
            timezone: timezone.clone(),
        })?;

        self.user_timezones.insert(user_id, timezone);
        Ok(())
    }

    /// Get user timezone preference
    #[must_use]
    pub fn get_user_timezone(&self, user_id: &str) -> Option<String> {
        self.user_timezones.get(user_id).cloned()
    }

    /// Format datetime in user's timezone
    pub fn format_for_user(
        &self,
        datetime: DateTime<Utc>,
        user_id: &str,
        format: TimeFormat,
    ) -> TimezoneResult<String> {
        let timezone = self
            .get_user_timezone(user_id)
            .unwrap_or_else(|| "UTC".to_string());

        let tz = Tz::from_str(&timezone).map_err(|_| TimezoneError::InvalidTimezone {
            timezone: timezone.clone(),
        })?;

        let local_time = datetime.with_timezone(&tz);

        let formatted = match format {
            TimeFormat::Hour12 => local_time.format("%I:%M %p").to_string(),
            TimeFormat::Hour24 => local_time.format("%H:%M").to_string(),
            TimeFormat::Iso8601 => local_time.to_rfc3339(),
            TimeFormat::Custom { format: fmt } => local_time.format(&fmt).to_string(),
        };

        Ok(formatted)
    }

    /// Calculate time until event in user's timezone
    pub fn time_until_event(
        &self,
        event_time: DateTime<Utc>,
        user_id: &str,
    ) -> TimezoneResult<String> {
        let timezone = self
            .get_user_timezone(user_id)
            .unwrap_or_else(|| "UTC".to_string());

        let tz = Tz::from_str(&timezone).map_err(|_| TimezoneError::InvalidTimezone {
            timezone: timezone.clone(),
        })?;

        let now = Utc::now().with_timezone(&tz);
        let event_local = event_time.with_timezone(&tz);

        let duration = event_local.signed_duration_since(now);

        Ok(self.format_duration(duration))
    }

    /// Find best meeting time across multiple timezones
    pub fn find_best_meeting_time(
        &self,
        participant_timezones: Vec<String>,
        duration_minutes: i64,
        working_hours_start: u32,
        working_hours_end: u32,
    ) -> TimezoneResult<Vec<DateTime<Utc>>> {
        let mut suggestions = Vec::new();
        let now = Utc::now();

        // Check next 7 days
        for day_offset in 0..7 {
            let check_date = now + Duration::days(day_offset);

            // Check every hour during working hours
            for hour in working_hours_start..working_hours_end {
                let mut all_in_working_hours = true;

                for tz_str in &participant_timezones {
                    let tz = Tz::from_str(tz_str).map_err(|_| TimezoneError::InvalidTimezone {
                        timezone: tz_str.clone(),
                    })?;

                    let time_in_tz = check_date.with_timezone(&tz);
                    let hour_in_tz = time_in_tz.hour();

                    if hour_in_tz < working_hours_start || hour_in_tz >= working_hours_end {
                        all_in_working_hours = false;
                        break;
                    }
                }

                if all_in_working_hours {
                    suggestions.push(check_date);
                    if suggestions.len() >= 5 {
                        return Ok(suggestions);
                    }
                }
            }
        }

        Ok(suggestions)
    }

    /// Get current time in multiple timezones
    pub fn get_world_clock(&self, timezones: Vec<String>) -> TimezoneResult<Vec<(String, String)>> {
        let now = Utc::now();
        let mut results = Vec::new();

        for tz_str in timezones {
            let tz = Tz::from_str(&tz_str).map_err(|_| TimezoneError::InvalidTimezone {
                timezone: tz_str.clone(),
            })?;

            let local_time = now.with_timezone(&tz);
            let formatted = local_time.format("%Y-%m-%d %H:%M:%S %Z").to_string();

            results.push((tz_str, formatted));
        }

        Ok(results)
    }

    /// Check if daylight saving time is active
    fn is_dst(&self, datetime: &DateTime<Tz>) -> bool {
        // Simplified DST detection
        let month = datetime.month();
        // Generally, DST is active in months 4-10 (April-October in Northern Hemisphere)
        (4..=10).contains(&month)
    }

    /// Format duration in human-readable format
    fn format_duration(&self, duration: Duration) -> String {
        let total_seconds = duration.num_seconds();

        if total_seconds < 0 {
            return "Past".to_string();
        }

        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;

        if days > 0 {
            format!("{days}d {hours}h")
        } else if hours > 0 {
            format!("{hours}h {minutes}m")
        } else if minutes > 0 {
            format!("{minutes}m")
        } else {
            "Now".to_string()
        }
    }

    /// List commonly used timezones
    #[must_use]
    pub fn get_common_timezones() -> Vec<(String, String)> {
        vec![
            (
                "America/New_York".to_string(),
                "Eastern Time (US & Canada)".to_string(),
            ),
            (
                "America/Chicago".to_string(),
                "Central Time (US & Canada)".to_string(),
            ),
            (
                "America/Denver".to_string(),
                "Mountain Time (US & Canada)".to_string(),
            ),
            (
                "America/Los_Angeles".to_string(),
                "Pacific Time (US & Canada)".to_string(),
            ),
            ("Europe/London".to_string(), "London".to_string()),
            (
                "Europe/Paris".to_string(),
                "Paris, Berlin, Rome".to_string(),
            ),
            ("Asia/Tokyo".to_string(), "Tokyo".to_string()),
            ("Asia/Shanghai".to_string(), "Beijing, Shanghai".to_string()),
            ("Asia/Kolkata".to_string(), "Mumbai, New Delhi".to_string()),
            (
                "Australia/Sydney".to_string(),
                "Sydney, Melbourne".to_string(),
            ),
            (
                "Pacific/Auckland".to_string(),
                "Auckland, Wellington".to_string(),
            ),
            ("UTC".to_string(), "Coordinated Universal Time".to_string()),
        ]
    }

    /// Detect timezone from offset
    #[must_use]
    pub fn detect_timezone_from_offset(&self, offset_minutes: i32) -> Vec<String> {
        let offset_seconds = offset_minutes * 60;
        let mut matches = Vec::new();

        // Common timezones
        let common = Self::get_common_timezones();
        for (tz_str, _) in common {
            if let Ok(tz) = Tz::from_str(&tz_str) {
                let now = Utc::now().with_timezone(&tz);
                if now.offset().fix().local_minus_utc() == offset_seconds {
                    matches.push(tz_str);
                }
            }
        }

        matches
    }
}

impl Default for TimezoneManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_timezone() {
        let manager = TimezoneManager::new();
        let utc_time = Utc::now();

        let result = manager.convert_timezone(utc_time, "America/New_York");
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_timezone() {
        let manager = TimezoneManager::new();
        let utc_time = Utc::now();

        let result = manager.convert_timezone(utc_time, "Invalid/Timezone");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_timezone_info() {
        let mut manager = TimezoneManager::new();

        let info = manager.get_timezone_info("America/New_York").unwrap();
        assert_eq!(info.identifier, "America/New_York");
        assert!(info.utc_offset_seconds != 0);
    }

    #[test]
    fn test_set_user_timezone() {
        let mut manager = TimezoneManager::new();

        manager
            .set_user_timezone("user1".to_string(), "Europe/London".to_string())
            .unwrap();

        let tz = manager.get_user_timezone("user1");
        assert_eq!(tz, Some("Europe/London".to_string()));
    }

    #[test]
    fn test_format_for_user() {
        let mut manager = TimezoneManager::new();
        manager
            .set_user_timezone("user1".to_string(), "America/New_York".to_string())
            .unwrap();

        let utc_time = Utc::now();
        let formatted = manager.format_for_user(utc_time, "user1", TimeFormat::Hour24);

        assert!(formatted.is_ok());
        let result = formatted.unwrap();
        assert!(result.contains(':'));
    }

    #[test]
    fn test_format_hour12() {
        let mut manager = TimezoneManager::new();
        manager
            .set_user_timezone("user1".to_string(), "UTC".to_string())
            .unwrap();

        let utc_time = Utc::now();
        let formatted = manager.format_for_user(utc_time, "user1", TimeFormat::Hour12);

        assert!(formatted.is_ok());
        let result = formatted.unwrap();
        assert!(result.contains("AM") || result.contains("PM"));
    }

    #[test]
    fn test_time_until_event() {
        let mut manager = TimezoneManager::new();
        manager
            .set_user_timezone("user1".to_string(), "UTC".to_string())
            .unwrap();

        let future_time = Utc::now() + Duration::hours(2);
        let result = manager.time_until_event(future_time, "user1").unwrap();

        assert!(result.contains("h") || result.contains("m"));
    }

    #[test]
    fn test_world_clock() {
        let manager = TimezoneManager::new();

        let timezones = vec![
            "America/New_York".to_string(),
            "Europe/London".to_string(),
            "Asia/Tokyo".to_string(),
        ];

        let result = manager.get_world_clock(timezones).unwrap();
        assert_eq!(result.len(), 3);

        for (tz, time_str) in result {
            assert!(!tz.is_empty());
            assert!(!time_str.is_empty());
        }
    }

    #[test]
    fn test_common_timezones() {
        let timezones = TimezoneManager::get_common_timezones();
        assert!(!timezones.is_empty());
        assert!(timezones.iter().any(|(id, _)| id == "America/New_York"));
        assert!(timezones.iter().any(|(id, _)| id == "Europe/London"));
    }

    #[test]
    fn test_detect_timezone_from_offset() {
        let manager = TimezoneManager::new();

        // Test UTC (0 offset)
        let utc_matches = manager.detect_timezone_from_offset(0);
        assert!(utc_matches.contains(&"UTC".to_string()));

        // Test EST (-5 hours)
        let est_matches = manager.detect_timezone_from_offset(-300);
        assert!(!est_matches.is_empty());
    }

    #[test]
    fn test_find_best_meeting_time() {
        let manager = TimezoneManager::new();

        let timezones = vec!["America/New_York".to_string(), "Europe/London".to_string()];

        let result = manager.find_best_meeting_time(timezones, 60, 9, 17);
        assert!(result.is_ok());
    }
}
