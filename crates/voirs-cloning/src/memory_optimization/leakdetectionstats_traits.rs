//! # LeakDetectionStats - Trait Implementations
//!
//! This module contains trait implementations for `LeakDetectionStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;
use std::time::SystemTime;

impl Default for LeakDetectionStats {
    fn default() -> Self {
        Self {
            tracked_allocations: 0,
            potential_leaks: 0,
            confirmed_leaks: 0,
            auto_cleaned_leaks: 0,
            total_leaked_bytes: 0,
            avg_allocation_lifetime: Duration::from_secs(0),
            pressure_events: 0,
            last_scan: SystemTime::now(),
            scan_duration_stats: DurationStats::default(),
        }
    }
}
