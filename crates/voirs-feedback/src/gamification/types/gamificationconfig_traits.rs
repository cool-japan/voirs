//! # `GamificationConfig` - Trait Implementations
//!
//! This module contains trait implementations for `GamificationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GamificationConfig;

impl Default for GamificationConfig {
    fn default() -> Self {
        Self {
            enable_achievements: true,
            enable_leaderboards: true,
            level_up_bonus: 50,
            streak_bonus_points: 25,
            max_leaderboard_size: 100,
        }
    }
}
