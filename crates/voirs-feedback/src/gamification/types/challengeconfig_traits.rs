//! # `ChallengeConfig` - Trait Implementations
//!
//! This module contains trait implementations for `ChallengeConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ChallengeConfig;

impl Default for ChallengeConfig {
    fn default() -> Self {
        Self {
            max_active_challenges: 5,
            challenge_refresh_days: 7,
            enable_time_limited_events: true,
            enable_community_challenges: true,
        }
    }
}
