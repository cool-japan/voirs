//! Gamification system for user engagement and motivation
//!
//! This module provides a comprehensive gamification framework including:
//!
//! - [`achievements`]: Achievement system with badges and progression tracking
//! - [`social`]: Social features including peer comparison and community
//! - [`points`]: Point system with multi-currency support and marketplace
//! - [`challenges`]: Challenge framework with time-limited events
//! - [`motivation`]: Motivation monitoring and intervention system
//! - [`leaderboards`]: Leaderboard system with rankings and tiers
//! - [`types`]: Common data structures and configurations

pub mod achievements;
pub mod challenges;
pub mod leaderboards;
pub mod motivation;
pub mod points;
pub mod social;
pub mod types;

// Re-export leaderboards explicitly (no conflicts)
pub use leaderboards::{
    Leaderboard, LeaderboardConfig, LeaderboardEntry, LeaderboardSummary, LeaderboardSystem,
    LeaderboardView,
};

// Note: achievements::*, challenges::*, motivation::*, points::*, social::* are not glob re-exported
// to avoid ambiguity due to duplicate type names (e.g., ChallengeStatus in both challenges and social).
// Import from specific modules: gamification::achievements::*, gamification::social::*, etc.
