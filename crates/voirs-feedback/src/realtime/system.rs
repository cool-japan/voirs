//! Core real-time feedback system

use super::stream::FeedbackStream;
use super::types::*;
use crate::traits::{FeedbackResponse, SessionState};
use crate::FeedbackError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::timeout;
use uuid::Uuid;
use voirs_sdk::AudioBuffer;

/// Main real-time feedback system
#[derive(Debug, Clone)]
pub struct RealtimeFeedbackSystem {
    /// System configuration
    config: RealtimeConfig,
    /// Active streams
    streams: Arc<RwLock<HashMap<Uuid, FeedbackStream>>>,
    /// System statistics
    stats: Arc<RwLock<RealtimeStats>>,
}

impl RealtimeFeedbackSystem {
    /// Create a new real-time feedback system
    pub async fn new() -> Result<Self, FeedbackError> {
        Self::with_config(RealtimeConfig::default()).await
    }

    /// Create a new real-time feedback system with custom configuration
    pub async fn with_config(config: RealtimeConfig) -> Result<Self, FeedbackError> {
        Ok(Self {
            config,
            streams: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RealtimeStats::default())),
        })
    }

    /// Create a new feedback stream
    pub async fn create_stream(
        &self,
        user_id: &str,
        session_state: &SessionState,
    ) -> Result<FeedbackStream, FeedbackError> {
        let stream = FeedbackStream::new(
            user_id.to_string(),
            self.config.clone(),
            session_state.clone(),
        );

        // Use non-blocking operations with timeout to preserve UI responsiveness
        let ui_timeout = Duration::from_millis(50);

        // Try to acquire both locks non-blocking for UI responsiveness
        let streams_result = self.streams.try_write();
        let stats_result = self.stats.try_write();

        match (streams_result, stats_result) {
            (Ok(mut streams), Ok(mut stats)) => {
                // Both locks acquired successfully - update atomically
                streams.insert(stream.stream_id, stream.clone());
                stats.active_streams += 1;
            }
            _ => {
                // One or both locks failed - return timeout to preserve UI responsiveness
                return Err(FeedbackError::Timeout);
            }
        }

        Ok(stream)
    }

    /// Get system statistics
    pub async fn get_statistics(&self) -> Result<RealtimeStats, FeedbackError> {
        // Use try_read for non-blocking access to preserve UI responsiveness
        match self.stats.try_read() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(FeedbackError::Timeout),
        }
    }

    /// Remove a stream
    pub async fn remove_stream(&self, stream_id: Uuid) -> Result<(), FeedbackError> {
        // Try to acquire both locks non-blocking for UI responsiveness
        let streams_result = self.streams.try_write();
        let stats_result = self.stats.try_write();

        match (streams_result, stats_result) {
            (Ok(mut streams), Ok(mut stats)) => {
                // Both locks acquired successfully - update atomically
                if streams.remove(&stream_id).is_some() {
                    // Only decrement stats if the stream actually existed
                    if stats.active_streams > 0 {
                        stats.active_streams -= 1;
                    }
                }
            }
            _ => {
                // One or both locks failed - return timeout to preserve UI responsiveness
                return Err(FeedbackError::Timeout);
            }
        }

        Ok(())
    }
}
