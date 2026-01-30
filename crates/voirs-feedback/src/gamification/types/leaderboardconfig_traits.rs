//! # `LeaderboardConfig` - Trait Implementations
//!
//! This module contains trait implementations for `LeaderboardConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LeaderboardConfig;

impl Default for LeaderboardConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            update_frequency_sec: 300,
            enable_realtime: true,
            anonymous_probability: 0.1,
            enable_fair_grouping: true,
            min_tier_participants: 10,
        }
    }
}
