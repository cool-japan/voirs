//! # `SocialConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SocialConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SocialConfig;

impl Default for SocialConfig {
    fn default() -> Self {
        Self {
            enable_peer_comparisons: true,
            enable_collaborative_challenges: true,
            enable_mentorship: true,
            enable_forums: true,
            max_peer_group_size: 10,
        }
    }
}
