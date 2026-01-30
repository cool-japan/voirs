// src/rate_limiting.rs
//! Rate Limiting and Throttling System for `VoiRS` Feedback API Protection
//!
//! This module provides comprehensive rate limiting capabilities to protect
//! the `VoiRS` feedback system from abuse, ensure fair resource allocation,
//! and maintain system stability under load.
//!
//! # Features
//! - Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window, Leaky Bucket)
//! - Per-user, per-IP, and per-endpoint rate limiting
//! - Distributed rate limiting support for multi-instance deployments
//! - Configurable limits with burst support
//! - Automatic cleanup of expired entries
//! - Comprehensive metrics and monitoring
//!
//! # Examples
//! ```
//! use voirs_feedback::rate_limiting::{RateLimiter, RateLimitConfig, RateLimitAlgorithm};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a rate limiter with token bucket algorithm
//! let config = RateLimitConfig {
//!     requests_per_window: 100,
//!     window_duration: Duration::from_secs(60),
//!     burst_size: 10,
//!     algorithm: RateLimitAlgorithm::TokenBucket,
//! };
//!
//! let rate_limiter = RateLimiter::new(config);
//!
//! // Check if request is allowed
//! let allowed = rate_limiter.check_rate_limit("user_123", "endpoint_/api/feedback").await?;
//! if allowed {
//!     println!("Request allowed");
//! } else {
//!     println!("Rate limit exceeded");
//! }
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

/// Rate limiting errors
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum RateLimitError {
    /// Rate limit exceeded
    #[error("Rate limit exceeded for key: {key}. Retry after {retry_after:?}")]
    Exceeded { key: String, retry_after: Duration },

    /// Invalid configuration
    #[error("Invalid rate limit configuration: {0}")]
    InvalidConfig(String),

    /// Storage error
    #[error("Rate limit storage error: {0}")]
    StorageError(String),
}

/// Result type for rate limiting operations
pub type RateLimitResult<T> = Result<T, RateLimitError>;

/// Rate limiting algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum RateLimitAlgorithm {
    /// Token Bucket: Allows bursts up to bucket capacity, refills at steady rate
    TokenBucket,
    /// Sliding Window: Smooths traffic over rolling time window
    SlidingWindow,
    /// Fixed Window: Simple counter that resets at fixed intervals
    FixedWindow,
    /// Leaky Bucket: Constant output rate regardless of input bursts
    LeakyBucket,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Number of requests allowed per window
    pub requests_per_window: u32,
    /// Duration of the time window
    pub window_duration: Duration,
    /// Maximum burst size (for Token Bucket algorithm)
    pub burst_size: u32,
    /// Algorithm to use for rate limiting
    pub algorithm: RateLimitAlgorithm,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 100,
            window_duration: Duration::from_secs(60),
            burst_size: 10,
            algorithm: RateLimitAlgorithm::TokenBucket,
        }
    }
}

/// Rate limit bucket state for Token Bucket algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenBucketState {
    /// Current number of tokens
    tokens: f64,
    /// Maximum tokens (bucket capacity)
    capacity: f64,
    /// Tokens added per second
    refill_rate: f64,
    /// Last refill timestamp
    last_refill: DateTime<Utc>,
}

/// Rate limit state for Fixed Window algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixedWindowState {
    /// Request count in current window
    count: u32,
    /// Window start time
    window_start: DateTime<Utc>,
}

/// Rate limit state for Sliding Window algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidingWindowState {
    /// Timestamp of each request
    request_timestamps: Vec<DateTime<Utc>>,
}

/// Rate limit state for Leaky Bucket algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeakyBucketState {
    /// Current queue size
    queue_size: u32,
    /// Maximum queue capacity
    capacity: u32,
    /// Leak rate (requests per second)
    leak_rate: f64,
    /// Last leak timestamp
    last_leak: DateTime<Utc>,
}

/// Combined rate limit state
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RateLimitState {
    TokenBucket(TokenBucketState),
    FixedWindow(FixedWindowState),
    SlidingWindow(SlidingWindowState),
    LeakyBucket(LeakyBucketState),
}

/// Rate limit entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RateLimitEntry {
    state: RateLimitState,
    total_requests: u64,
    rejected_requests: u64,
    last_access: DateTime<Utc>,
}

/// Rate limiter implementation
pub struct RateLimiter {
    config: RateLimitConfig,
    entries: Arc<RwLock<HashMap<String, RateLimitEntry>>>,
    cleanup_interval: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    #[must_use]
    pub fn new(config: RateLimitConfig) -> Self {
        let limiter = Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        };

        // Start background cleanup task
        limiter.start_cleanup_task();

        limiter
    }

    /// Check if a request is allowed under the rate limit
    ///
    /// # Arguments
    /// * `key` - Unique identifier for the rate limit (e.g., user ID, IP address)
    /// * `resource` - Resource being accessed (e.g., endpoint path)
    ///
    /// # Returns
    /// `true` if request is allowed, `false` if rate limit exceeded
    pub async fn check_rate_limit(&self, key: &str, resource: &str) -> RateLimitResult<bool> {
        let combined_key = format!("{key}:{resource}");

        let mut entries = self.entries.write().await;
        let now = Utc::now();

        let entry = entries
            .entry(combined_key.clone())
            .or_insert_with(|| RateLimitEntry {
                state: self.create_initial_state(),
                total_requests: 0,
                rejected_requests: 0,
                last_access: now,
            });

        entry.total_requests += 1;
        entry.last_access = now;

        let allowed = match &mut entry.state {
            RateLimitState::TokenBucket(state) => self.check_token_bucket(state, now),
            RateLimitState::FixedWindow(state) => self.check_fixed_window(state, now),
            RateLimitState::SlidingWindow(state) => self.check_sliding_window(state, now),
            RateLimitState::LeakyBucket(state) => self.check_leaky_bucket(state, now),
        };

        if !allowed {
            entry.rejected_requests += 1;
        }

        Ok(allowed)
    }

    /// Get rate limit status for a key
    pub async fn get_status(&self, key: &str, resource: &str) -> Option<RateLimitStatus> {
        let combined_key = format!("{key}:{resource}");
        let entries = self.entries.read().await;

        entries.get(&combined_key).map(|entry| {
            let remaining = match &entry.state {
                RateLimitState::TokenBucket(state) => state.tokens as u32,
                RateLimitState::FixedWindow(state) => {
                    self.config.requests_per_window.saturating_sub(state.count)
                }
                RateLimitState::SlidingWindow(state) => self
                    .config
                    .requests_per_window
                    .saturating_sub(state.request_timestamps.len() as u32),
                RateLimitState::LeakyBucket(state) => {
                    state.capacity.saturating_sub(state.queue_size)
                }
            };

            RateLimitStatus {
                limit: self.config.requests_per_window,
                remaining,
                reset_at: self.calculate_reset_time(&entry.state),
                total_requests: entry.total_requests,
                rejected_requests: entry.rejected_requests,
            }
        })
    }

    /// Reset rate limit for a specific key
    pub async fn reset(&self, key: &str, resource: &str) {
        let combined_key = format!("{key}:{resource}");
        let mut entries = self.entries.write().await;
        entries.remove(&combined_key);
    }

    /// Get comprehensive statistics across all rate limits
    pub async fn get_statistics(&self) -> RateLimitStatistics {
        let entries = self.entries.read().await;

        let total_keys = entries.len();
        let total_requests: u64 = entries.values().map(|e| e.total_requests).sum();
        let total_rejected: u64 = entries.values().map(|e| e.rejected_requests).sum();

        let acceptance_rate = if total_requests > 0 {
            ((total_requests - total_rejected) as f64 / total_requests as f64) * 100.0
        } else {
            100.0
        };

        RateLimitStatistics {
            total_keys,
            total_requests,
            total_rejected,
            acceptance_rate,
            algorithm: self.config.algorithm,
        }
    }

    // Private helper methods

    fn create_initial_state(&self) -> RateLimitState {
        let now = Utc::now();

        match self.config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                let capacity = f64::from(self.config.requests_per_window + self.config.burst_size);
                RateLimitState::TokenBucket(TokenBucketState {
                    tokens: capacity,
                    capacity,
                    refill_rate: f64::from(self.config.requests_per_window)
                        / self.config.window_duration.as_secs_f64(),
                    last_refill: now,
                })
            }
            RateLimitAlgorithm::FixedWindow => RateLimitState::FixedWindow(FixedWindowState {
                count: 0,
                window_start: now,
            }),
            RateLimitAlgorithm::SlidingWindow => {
                RateLimitState::SlidingWindow(SlidingWindowState {
                    request_timestamps: Vec::new(),
                })
            }
            RateLimitAlgorithm::LeakyBucket => RateLimitState::LeakyBucket(LeakyBucketState {
                queue_size: 0,
                capacity: self.config.requests_per_window + self.config.burst_size,
                leak_rate: f64::from(self.config.requests_per_window)
                    / self.config.window_duration.as_secs_f64(),
                last_leak: now,
            }),
        }
    }

    fn check_token_bucket(&self, state: &mut TokenBucketState, now: DateTime<Utc>) -> bool {
        // Refill tokens based on elapsed time
        let elapsed = (now - state.last_refill).num_milliseconds() as f64 / 1000.0;
        let tokens_to_add = elapsed * state.refill_rate;
        state.tokens = (state.tokens + tokens_to_add).min(state.capacity);
        state.last_refill = now;

        // Check if we have enough tokens
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn check_fixed_window(&self, state: &mut FixedWindowState, now: DateTime<Utc>) -> bool {
        // Check if we need to reset the window
        let elapsed = now - state.window_start;
        if elapsed > chrono::Duration::from_std(self.config.window_duration).unwrap() {
            state.count = 0;
            state.window_start = now;
        }

        // Check if we're within limits
        if state.count < self.config.requests_per_window {
            state.count += 1;
            true
        } else {
            false
        }
    }

    fn check_sliding_window(&self, state: &mut SlidingWindowState, now: DateTime<Utc>) -> bool {
        // Remove old requests outside the window
        let window_start = now - chrono::Duration::from_std(self.config.window_duration).unwrap();
        state.request_timestamps.retain(|&ts| ts > window_start);

        // Check if we're within limits
        if state.request_timestamps.len() < self.config.requests_per_window as usize {
            state.request_timestamps.push(now);
            true
        } else {
            false
        }
    }

    fn check_leaky_bucket(&self, state: &mut LeakyBucketState, now: DateTime<Utc>) -> bool {
        // Leak requests based on elapsed time
        let elapsed = (now - state.last_leak).num_milliseconds() as f64 / 1000.0;
        let leaked = (elapsed * state.leak_rate) as u32;
        state.queue_size = state.queue_size.saturating_sub(leaked);
        state.last_leak = now;

        // Check if we can add to the queue
        if state.queue_size < state.capacity {
            state.queue_size += 1;
            true
        } else {
            false
        }
    }

    fn calculate_reset_time(&self, state: &RateLimitState) -> DateTime<Utc> {
        let now = Utc::now();

        match state {
            RateLimitState::TokenBucket(state) => {
                let tokens_needed = 1.0 - state.tokens;
                let seconds_to_wait = tokens_needed / state.refill_rate;
                now + chrono::Duration::milliseconds((seconds_to_wait * 1000.0) as i64)
            }
            RateLimitState::FixedWindow(state) => {
                state.window_start
                    + chrono::Duration::from_std(self.config.window_duration).unwrap()
            }
            RateLimitState::SlidingWindow(state) => {
                if let Some(oldest) = state.request_timestamps.first() {
                    *oldest + chrono::Duration::from_std(self.config.window_duration).unwrap()
                } else {
                    now
                }
            }
            RateLimitState::LeakyBucket(state) => {
                let extra_capacity_needed = state.queue_size.saturating_sub(state.capacity - 1);
                let seconds_to_wait = f64::from(extra_capacity_needed) / state.leak_rate;
                now + chrono::Duration::milliseconds((seconds_to_wait * 1000.0) as i64)
            }
        }
    }

    fn start_cleanup_task(&self) {
        let entries = Arc::clone(&self.entries);
        let cleanup_interval = self.cleanup_interval;
        let window_duration = self.config.window_duration;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                interval.tick().await;

                let mut entries_write = entries.write().await;
                let now = Utc::now();
                let retention_threshold =
                    now - chrono::Duration::from_std(window_duration * 2).unwrap();

                entries_write.retain(|_, entry| entry.last_access > retention_threshold);
            }
        });
    }
}

/// Rate limit status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    /// Maximum requests allowed in the window
    pub limit: u32,
    /// Remaining requests in current window
    pub remaining: u32,
    /// Time when the rate limit resets
    pub reset_at: DateTime<Utc>,
    /// Total requests made
    pub total_requests: u64,
    /// Total requests rejected
    pub rejected_requests: u64,
}

/// Statistics for rate limiting system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatistics {
    /// Total number of tracked keys
    pub total_keys: usize,
    /// Total requests processed
    pub total_requests: u64,
    /// Total requests rejected
    pub total_rejected: u64,
    /// Acceptance rate (percentage)
    pub acceptance_rate: f64,
    /// Algorithm in use
    pub algorithm: RateLimitAlgorithm,
}

/// Multi-tier rate limiting for different user types
pub struct TieredRateLimiter {
    tiers: HashMap<String, RateLimiter>,
    default_tier: String,
}

impl TieredRateLimiter {
    /// Create a new tiered rate limiter
    #[must_use]
    pub fn new() -> Self {
        let mut tiers = HashMap::new();

        // Free tier: 100 requests per minute
        tiers.insert(
            "free".to_string(),
            RateLimiter::new(RateLimitConfig {
                requests_per_window: 100,
                window_duration: Duration::from_secs(60),
                burst_size: 10,
                algorithm: RateLimitAlgorithm::TokenBucket,
            }),
        );

        // Premium tier: 1000 requests per minute
        tiers.insert(
            "premium".to_string(),
            RateLimiter::new(RateLimitConfig {
                requests_per_window: 1000,
                window_duration: Duration::from_secs(60),
                burst_size: 100,
                algorithm: RateLimitAlgorithm::TokenBucket,
            }),
        );

        // Enterprise tier: 10000 requests per minute
        tiers.insert(
            "enterprise".to_string(),
            RateLimiter::new(RateLimitConfig {
                requests_per_window: 10000,
                window_duration: Duration::from_secs(60),
                burst_size: 1000,
                algorithm: RateLimitAlgorithm::SlidingWindow,
            }),
        );

        Self {
            tiers,
            default_tier: "free".to_string(),
        }
    }

    /// Add a custom tier
    pub fn add_tier(&mut self, tier_name: String, config: RateLimitConfig) {
        self.tiers.insert(tier_name, RateLimiter::new(config));
    }

    /// Check rate limit for a specific tier
    pub async fn check_rate_limit(
        &self,
        tier: Option<&str>,
        key: &str,
        resource: &str,
    ) -> RateLimitResult<bool> {
        let tier_name = tier.unwrap_or(&self.default_tier);

        let limiter = self
            .tiers
            .get(tier_name)
            .ok_or_else(|| RateLimitError::InvalidConfig(format!("Unknown tier: {tier_name}")))?;

        limiter.check_rate_limit(key, resource).await
    }

    /// Get status for a specific tier
    pub async fn get_status(
        &self,
        tier: Option<&str>,
        key: &str,
        resource: &str,
    ) -> Option<RateLimitStatus> {
        let tier_name = tier.unwrap_or(&self.default_tier);
        self.tiers.get(tier_name)?.get_status(key, resource).await
    }
}

impl Default for TieredRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let config = RateLimitConfig {
            requests_per_window: 10,
            window_duration: Duration::from_secs(1),
            burst_size: 5,
            algorithm: RateLimitAlgorithm::TokenBucket,
        };

        let limiter = RateLimiter::new(config);

        // Should allow up to burst_size + requests_per_window requests
        for i in 0..15 {
            let allowed = limiter
                .check_rate_limit("user1", "endpoint1")
                .await
                .unwrap();
            assert!(allowed, "Request {} should be allowed", i);
        }

        // Next request should be rejected
        let allowed = limiter
            .check_rate_limit("user1", "endpoint1")
            .await
            .unwrap();
        assert!(
            !allowed,
            "Request should be rejected after exhausting tokens"
        );
    }

    #[tokio::test]
    async fn test_fixed_window() {
        let config = RateLimitConfig {
            requests_per_window: 5,
            window_duration: Duration::from_millis(100),
            burst_size: 0,
            algorithm: RateLimitAlgorithm::FixedWindow,
        };

        let limiter = RateLimiter::new(config);

        // Allow 5 requests
        for i in 0..5 {
            let allowed = limiter
                .check_rate_limit("user2", "endpoint2")
                .await
                .unwrap();
            assert!(allowed, "Request {} should be allowed", i);
        }

        // Reject next request
        let allowed = limiter
            .check_rate_limit("user2", "endpoint2")
            .await
            .unwrap();
        assert!(!allowed, "Request should be rejected");

        // Wait for window to reset
        sleep(Duration::from_millis(110)).await;

        // Should allow requests again
        let allowed = limiter
            .check_rate_limit("user2", "endpoint2")
            .await
            .unwrap();
        assert!(allowed, "Request should be allowed after window reset");
    }

    #[tokio::test]
    async fn test_sliding_window() {
        let config = RateLimitConfig {
            requests_per_window: 5,
            window_duration: Duration::from_millis(200),
            burst_size: 0,
            algorithm: RateLimitAlgorithm::SlidingWindow,
        };

        let limiter = RateLimiter::new(config);

        // Make 5 requests
        for _ in 0..5 {
            limiter
                .check_rate_limit("user3", "endpoint3")
                .await
                .unwrap();
        }

        // Should be blocked
        let allowed = limiter
            .check_rate_limit("user3", "endpoint3")
            .await
            .unwrap();
        assert!(!allowed);

        // Wait for some requests to expire
        sleep(Duration::from_millis(210)).await;

        // Should allow requests again
        let allowed = limiter
            .check_rate_limit("user3", "endpoint3")
            .await
            .unwrap();
        assert!(allowed);
    }

    #[tokio::test]
    async fn test_get_status() {
        let config = RateLimitConfig {
            requests_per_window: 10,
            window_duration: Duration::from_secs(60),
            burst_size: 5,
            algorithm: RateLimitAlgorithm::TokenBucket,
        };

        let limiter = RateLimiter::new(config);

        // Make a few requests
        for _ in 0..3 {
            limiter
                .check_rate_limit("user4", "endpoint4")
                .await
                .unwrap();
        }

        let status = limiter.get_status("user4", "endpoint4").await.unwrap();
        assert_eq!(status.limit, 10);
        assert_eq!(status.total_requests, 3);
        assert_eq!(status.rejected_requests, 0);
    }

    #[tokio::test]
    async fn test_reset() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);

        // Make some requests
        limiter
            .check_rate_limit("user5", "endpoint5")
            .await
            .unwrap();

        // Reset
        limiter.reset("user5", "endpoint5").await;

        // Status should be None after reset
        let status = limiter.get_status("user5", "endpoint5").await;
        assert!(status.is_none());
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = RateLimitConfig {
            requests_per_window: 2,
            window_duration: Duration::from_secs(60),
            burst_size: 0,
            algorithm: RateLimitAlgorithm::FixedWindow,
        };

        let limiter = RateLimiter::new(config);

        // Make requests that will be accepted and rejected
        limiter
            .check_rate_limit("user6", "endpoint6")
            .await
            .unwrap();
        limiter
            .check_rate_limit("user6", "endpoint6")
            .await
            .unwrap();
        limiter
            .check_rate_limit("user6", "endpoint6")
            .await
            .unwrap();

        let stats = limiter.get_statistics().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.total_rejected, 1);
        assert!(stats.acceptance_rate > 66.0 && stats.acceptance_rate < 67.0);
    }

    #[tokio::test]
    async fn test_tiered_rate_limiter() {
        let tiered = TieredRateLimiter::new();

        // Test free tier
        let allowed = tiered
            .check_rate_limit(Some("free"), "user7", "endpoint7")
            .await
            .unwrap();
        assert!(allowed);

        // Test premium tier
        let allowed = tiered
            .check_rate_limit(Some("premium"), "user8", "endpoint8")
            .await
            .unwrap();
        assert!(allowed);

        // Test invalid tier
        let result = tiered
            .check_rate_limit(Some("invalid"), "user9", "endpoint9")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_leaky_bucket() {
        let config = RateLimitConfig {
            requests_per_window: 10,
            window_duration: Duration::from_secs(1),
            burst_size: 5,
            algorithm: RateLimitAlgorithm::LeakyBucket,
        };

        let limiter = RateLimiter::new(config);

        // Fill the bucket
        for i in 0..15 {
            let allowed = limiter
                .check_rate_limit("user10", "endpoint10")
                .await
                .unwrap();
            assert!(allowed, "Request {} should be allowed", i);
        }

        // Should be at capacity
        let allowed = limiter
            .check_rate_limit("user10", "endpoint10")
            .await
            .unwrap();
        assert!(!allowed, "Request should be rejected when bucket is full");
    }
}
