//! # `PointSystemConfig` - Trait Implementations
//!
//! This module contains trait implementations for `PointSystemConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PointSystemConfig;

impl Default for PointSystemConfig {
    fn default() -> Self {
        Self {
            base_points_per_session: 10,
            streak_bonus_multiplier: 1.5,
            enable_marketplace: true,
            enable_transfers: true,
            max_daily_points: 1000,
        }
    }
}
