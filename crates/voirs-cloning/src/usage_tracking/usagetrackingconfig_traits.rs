//! # UsageTrackingConfig - Trait Implementations
//!
//! This module contains trait implementations for `UsageTrackingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for UsageTrackingConfig {
    fn default() -> Self {
        UsageTrackingConfig {
            enable_tracking: true,
            track_resource_usage: true,
            track_quality_metrics: true,
            track_security_events: true,
            retention_days: 365,
            max_records_in_memory: 10000,
            batch_size: 100,
            flush_interval: Duration::from_secs(60),
            anonymize_user_data: true,
            encrypt_sensitive_data: true,
        }
    }
}
