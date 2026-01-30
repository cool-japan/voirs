//! Log filtering capabilities.
//!
//! This module provides filtering mechanisms to control which log messages
//! are output based on various criteria.

use super::{Arc, LogEntry, LogLevel, RwLock};
use chrono::Utc;
use regex::Regex;
use std::collections::HashMap;

/// Log filter trait
pub trait LogFilter: Send + Sync {
    /// Check if a log entry should be output
    fn should_log(&self, entry: &LogEntry) -> bool;
}

/// Filter by log level
pub struct LevelFilter {
    min_level: LogLevel,
}

impl LevelFilter {
    /// Create new level filter
    #[must_use]
    pub fn new(min_level: LogLevel) -> Self {
        Self { min_level }
    }
}

impl LogFilter for LevelFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        entry.level >= self.min_level
    }
}

/// Filter by module path
pub struct ModuleFilter {
    allowed_modules: Vec<String>,
}

impl ModuleFilter {
    /// Create new module filter
    #[must_use]
    pub fn new(allowed_modules: Vec<String>) -> Self {
        Self { allowed_modules }
    }
}

impl LogFilter for ModuleFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if self.allowed_modules.is_empty() {
            return true;
        }

        entry.module.as_ref().is_some_and(|m| {
            self.allowed_modules
                .iter()
                .any(|allowed| m.starts_with(allowed))
        })
    }
}

/// Filter by message content
pub struct ContentFilter {
    pattern: Regex,
    exclude: bool,
}

impl ContentFilter {
    /// Create new content filter
    pub fn new(pattern: &str, exclude: bool) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            exclude,
        })
    }

    /// Create include filter
    pub fn include(pattern: &str) -> Result<Self, regex::Error> {
        Self::new(pattern, false)
    }

    /// Create exclude filter
    pub fn exclude(pattern: &str) -> Result<Self, regex::Error> {
        Self::new(pattern, true)
    }
}

impl LogFilter for ContentFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        let matches = self.pattern.is_match(&entry.message);
        if self.exclude {
            !matches
        } else {
            matches
        }
    }
}

/// Composite filter combining multiple filters
pub struct CompositeFilter {
    filters: Vec<Box<dyn LogFilter>>,
    match_all: bool,
}

impl CompositeFilter {
    /// Create new composite filter
    #[must_use]
    pub fn new(match_all: bool) -> Self {
        Self {
            filters: Vec::new(),
            match_all,
        }
    }

    /// Add a filter
    #[must_use]
    pub fn add_filter(mut self, filter: Box<dyn LogFilter>) -> Self {
        self.filters.push(filter);
        self
    }
}

impl LogFilter for CompositeFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if self.filters.is_empty() {
            return true;
        }

        if self.match_all {
            self.filters.iter().all(|f| f.should_log(entry))
        } else {
            self.filters.iter().any(|f| f.should_log(entry))
        }
    }
}

/// Rate limiting filter
pub struct RateLimitFilter {
    max_per_second: usize,
    window_start: Arc<RwLock<std::time::Instant>>,
    count: Arc<RwLock<usize>>,
}

impl RateLimitFilter {
    /// Create new rate limit filter
    #[must_use]
    pub fn new(max_per_second: usize) -> Self {
        Self {
            max_per_second,
            window_start: Arc::new(RwLock::new(std::time::Instant::now())),
            count: Arc::new(RwLock::new(0)),
        }
    }
}

impl LogFilter for RateLimitFilter {
    fn should_log(&self, _entry: &LogEntry) -> bool {
        let now = std::time::Instant::now();
        let mut window_start = self.window_start.write();
        let mut count = self.count.write();

        // Reset if window expired
        if now.duration_since(*window_start).as_secs() >= 1 {
            *window_start = now;
            *count = 0;
        }

        // Check rate limit
        if *count < self.max_per_second {
            *count += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(level: LogLevel, message: &str) -> LogEntry {
        LogEntry {
            level,
            timestamp: Utc::now(),
            message: message.to_string(),
            module: Some("test::module".to_string()),
            source: None,
            thread_id: None,
            context: HashMap::new(),
            metrics: None,
        }
    }

    #[test]
    fn test_level_filter() {
        let filter = LevelFilter::new(LogLevel::Warn);

        let info_entry = create_test_entry(LogLevel::Info, "Info message");
        let warn_entry = create_test_entry(LogLevel::Warn, "Warning message");
        let error_entry = create_test_entry(LogLevel::Error, "Error message");

        assert!(!filter.should_log(&info_entry));
        assert!(filter.should_log(&warn_entry));
        assert!(filter.should_log(&error_entry));
    }

    #[test]
    fn test_module_filter() {
        let filter = ModuleFilter::new(vec!["test".to_string(), "auth".to_string()]);

        let test_entry = create_test_entry(LogLevel::Info, "Test message");
        let mut other_entry = create_test_entry(LogLevel::Info, "Other message");
        other_entry.module = Some("other::module".to_string());

        assert!(filter.should_log(&test_entry));
        assert!(!filter.should_log(&other_entry));
    }

    #[test]
    fn test_content_filter_include() {
        let filter = ContentFilter::include(r"error|fail").unwrap();

        let error_entry = create_test_entry(LogLevel::Info, "An error occurred");
        let success_entry = create_test_entry(LogLevel::Info, "Operation succeeded");

        assert!(filter.should_log(&error_entry));
        assert!(!filter.should_log(&success_entry));
    }

    #[test]
    fn test_content_filter_exclude() {
        let filter = ContentFilter::exclude(r"(?i)debug|trace").unwrap(); // Case-insensitive

        let debug_entry = create_test_entry(LogLevel::Info, "debug information");
        let normal_entry = create_test_entry(LogLevel::Info, "Normal message");

        assert!(!filter.should_log(&debug_entry));
        assert!(filter.should_log(&normal_entry));
    }

    #[test]
    fn test_composite_filter() {
        let level_filter = Box::new(LevelFilter::new(LogLevel::Info));
        let content_filter = Box::new(ContentFilter::include(r"important").unwrap());

        let filter = CompositeFilter::new(true)
            .add_filter(level_filter)
            .add_filter(content_filter);

        let matching_entry = create_test_entry(LogLevel::Info, "This is important");
        let wrong_level = create_test_entry(LogLevel::Debug, "This is important");
        let wrong_content = create_test_entry(LogLevel::Info, "This is trivial");

        assert!(filter.should_log(&matching_entry));
        assert!(!filter.should_log(&wrong_level));
        assert!(!filter.should_log(&wrong_content));
    }

    #[test]
    fn test_rate_limit_filter() {
        let filter = RateLimitFilter::new(2);
        let entry = create_test_entry(LogLevel::Info, "Test");

        // First two should pass
        assert!(filter.should_log(&entry));
        assert!(filter.should_log(&entry));

        // Third should be rate limited
        assert!(!filter.should_log(&entry));

        // Wait for window to reset
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // Should work again
        assert!(filter.should_log(&entry));
    }
}
