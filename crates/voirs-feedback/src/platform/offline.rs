//! Offline capability support for `VoiRS` feedback system
//!
//! This module provides offline functionality including data caching, offline-first
//! operations, and seamless online/offline transitions.

use crate::traits::{FeedbackResponse, SessionState, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Real, bounded network-reachability probe: attempts a short-timeout TCP
/// connect to `endpoint` (a `"host:port"` string). A bare TCP handshake is
/// enough to establish real reachability -- no application data is ever
/// sent. Returns `false` (offline) for an unparseable endpoint, a
/// connection refusal, or a timeout; never panics or blocks the *caller*
/// longer than `timeout`.
///
/// `endpoint` may name a host by IP literal or by hostname. Hostname
/// resolution (`ToSocketAddrs::to_socket_addrs`) is a blocking OS call
/// with **no** timeout of its own -- `TcpStream::connect_timeout` only
/// bounds the connect step that follows a successful lookup, not the
/// lookup itself. A hung or slow resolver could otherwise block this
/// (synchronous) function, and therefore any async caller that invokes it
/// without `spawn_blocking`, far longer than `timeout`. To keep the bound
/// honest regardless of hostname vs. IP, the real lookup+connect runs on a
/// detached thread and the wait for its answer is what is bounded by
/// `timeout`; a resolver that is still hung when the deadline passes
/// leaves that thread running in the background (it exits on its own once
/// the OS-level resolution eventually completes or errors) and this
/// function honestly reports `false` ("not confirmed reachable in time").
///
/// `pub(super)`: also reused by [`super::sync::SyncManager::check_network_availability`]
/// for the same real-connectivity-check purpose.
pub(super) fn probe_connectivity(endpoint: &str, timeout: Duration) -> bool {
    use std::net::ToSocketAddrs;
    use std::sync::mpsc;

    let endpoint = endpoint.to_string();
    let (result_tx, result_rx) = mpsc::channel();
    let _ = std::thread::spawn(move || {
        let reachable = (|| {
            let mut addrs = endpoint.to_socket_addrs().ok()?;
            let addr = addrs.next()?;
            Some(std::net::TcpStream::connect_timeout(&addr, timeout).is_ok())
        })()
        .unwrap_or(false);
        // Best-effort: if the caller already timed out and stopped
        // listening, the receiver is gone and there is nobody left to
        // deliver this result to.
        let _ = result_tx.send(reachable);
    });

    result_rx.recv_timeout(timeout).unwrap_or(false)
}

/// Real outcome of an [`OfflineManager::sync_offline_operations`] pass.
#[derive(Debug, Clone, Copy, Default)]
struct SyncReport {
    /// Operations genuinely confirmed synced and removed from the queue.
    synced: usize,
    /// Operations that were not confirmed synced this pass -- either still
    /// queued for retry, or permanently dropped after exhausting retries.
    /// Either way, this is real data that did not reach the server.
    failed: usize,
}

/// Offline manager for handling offline operations
pub struct OfflineManager {
    config: OfflineConfig,
    cache: OfflineCache,
    queue: OperationQueue,
    /// Description
    pub storage: OfflineStorage,
    /// Cached result of the last real connectivity probe (see
    /// [`OfflineManager::is_offline`]), so repeated calls within
    /// `config.connectivity_cache_ttl_ms` do not each pay for a fresh TCP
    /// handshake.
    connectivity_cache: Mutex<Option<(Instant, bool)>>,
}

impl OfflineManager {
    /// Create a new offline manager. `config.storage_directory` is used as
    /// the real on-disk root for both cached model/data files (under
    /// `storage_directory/models`) and persisted user progress.
    #[must_use]
    pub fn new(config: OfflineConfig) -> Self {
        let models_dir = config.storage_directory.join("models");
        let storage = OfflineStorage::with_directory(config.storage_directory.clone());
        Self {
            cache: OfflineCache::with_models_dir(models_dir),
            queue: OperationQueue::new(),
            storage,
            connectivity_cache: Mutex::new(None),
            config,
        }
    }

    /// Check if the system is currently offline.
    ///
    /// Backed by a real, bounded network-reachability probe -- a
    /// short-timeout TCP connect to `config.connectivity_check_endpoint`
    /// (see [`probe_connectivity`]) -- not a hardcoded constant. The real
    /// result is cached for `config.connectivity_cache_ttl_ms` so repeated
    /// calls do not each block on a fresh handshake.
    #[must_use]
    pub fn is_offline(&self) -> bool {
        !self.is_online()
    }

    /// Real, cached connectivity check backing [`OfflineManager::is_offline`].
    fn is_online(&self) -> bool {
        let ttl = Duration::from_millis(self.config.connectivity_cache_ttl_ms);

        {
            let cached = self
                .connectivity_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some((checked_at, online)) = *cached {
                if checked_at.elapsed() < ttl {
                    return online;
                }
            }
        }

        let timeout = Duration::from_millis(self.config.connectivity_check_timeout_ms);
        let online = probe_connectivity(&self.config.connectivity_check_endpoint, timeout);

        let mut cached = self
            .connectivity_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *cached = Some((Instant::now(), online));
        online
    }

    /// Switch to offline mode
    pub fn switch_to_offline(&mut self) -> Result<(), OfflineError> {
        // Prepare system for offline operation
        self.cache.preload_essential_data()?;
        self.storage.prepare_offline_storage()?;
        Ok(())
    }

    /// Switch to online mode.
    ///
    /// Fails closed (`Err`) if any queued operation could not actually be
    /// delivered by the real sync attempt below -- e.g. no sync endpoint is
    /// configured, or the server rejected/was unreachable for the request
    /// -- rather than silently discarding it while reporting success.
    pub async fn switch_to_online(&mut self) -> Result<(), OfflineError> {
        // Sync offline operations
        let report = self.sync_offline_operations().await?;

        if report.failed > 0 {
            return Err(OfflineError::NetworkError {
                message: format!(
                    "{} of {} queued offline operation(s) could not be synced to the server",
                    report.failed,
                    report.synced + report.failed
                ),
            });
        }

        // Clear old cache if needed
        if self.config.clear_cache_on_online {
            self.cache.clear_expired_data()?;
        }

        Ok(())
    }

    /// Process feedback in offline mode
    pub async fn process_feedback_offline(
        &mut self,
        session: &SessionState,
        audio_data: &[u8],
        text: &str,
    ) -> Result<FeedbackResponse, OfflineError> {
        // Check if we have cached models
        if !self.cache.has_cached_models() {
            return Err(OfflineError::ModelNotCached);
        }

        // Process using cached models
        let feedback = self.generate_offline_feedback(session, audio_data, text)?;

        // Queue for later sync
        let operation = QueuedOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: OperationType::ProcessFeedback,
            session_id: session.session_id.to_string(),
            data: serde_json::to_value(&feedback).map_err(|e| {
                OfflineError::SerializationError {
                    message: format!("failed to serialize offline feedback for sync queue: {e}"),
                }
            })?,
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        self.queue.add_operation(operation);

        Ok(feedback)
    }

    /// Generate feedback using cached models.
    ///
    /// This is a lightweight, fully offline heuristic derived from the real
    /// input -- not a neural pronunciation grader. The cached model files
    /// gated by `has_cached_models()` in the caller prove real model
    /// *artifacts* are present on disk, but this code path does not
    /// actually execute one (no inference runtime ships with this offline
    /// path). Instead, real DSP statistics are computed directly from
    /// `audio_data` -- interpreted as little-endian 16-bit PCM samples,
    /// `VoiRS`'s standard in-memory audio representation -- so the output
    /// genuinely varies with the real input instead of being a constant.
    /// The measured quantities are exposed in `UserFeedback::metadata` for
    /// auditability rather than laundered into an opaque score.
    fn generate_offline_feedback(
        &self,
        _session: &SessionState,
        audio_data: &[u8],
        text: &str,
    ) -> Result<FeedbackResponse, OfflineError> {
        let start = Instant::now();

        if audio_data.is_empty() {
            return Err(OfflineError::InvalidInput {
                message: "cannot generate offline feedback: audio_data is empty".to_string(),
            });
        }

        let samples: Vec<i16> = audio_data
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let sample_count = samples.len().max(1);

        // Mean absolute amplitude, normalized against the full i16 range --
        // a real (if simple) proxy for signal energy.
        let sum_abs: u64 = samples.iter().map(|&s| u64::from(s.unsigned_abs())).sum();
        let mean_abs = sum_abs as f64 / sample_count as f64;
        let amplitude_ratio = (mean_abs / f64::from(i16::MAX)).clamp(0.0, 1.0);

        // Fraction of near-silent samples. A fixed, documented noise-floor
        // threshold -- not tuned against any real corpus.
        const SILENCE_THRESHOLD: u16 = 200;
        let silent_samples = samples
            .iter()
            .filter(|&&s| s.unsigned_abs() < SILENCE_THRESHOLD)
            .count();
        let silence_ratio = f64::from(u32::try_from(silent_samples).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(sample_count).unwrap_or(u32::MAX));
        let word_count = text.split_whitespace().count();

        // Real, input-dependent [0, 1] signal-presence score: more energy
        // and less silence both push it up. Never a substitute for a real
        // pronunciation model -- see the doc comment above.
        let signal_score =
            ((amplitude_ratio * 4.0).min(1.0) * (1.0 - silence_ratio)).clamp(0.0, 1.0);
        let overall_score = (0.2 + 0.8 * signal_score) as f32;
        // More real samples to average the amplitude/silence estimate over
        // means more confidence in *that estimate* -- capped well short of
        // 1.0 since this remains a coarse heuristic.
        let confidence = (0.3 + 0.5 * (sample_count as f32 / 16_000.0).min(1.0)).clamp(0.0, 0.8);

        let mut metadata = HashMap::new();
        metadata.insert("heuristic".to_string(), "offline_signal_v1".to_string());
        metadata.insert("sample_count".to_string(), sample_count.to_string());
        metadata.insert("mean_abs_amplitude".to_string(), format!("{mean_abs:.2}"));
        metadata.insert("silence_ratio".to_string(), format!("{silence_ratio:.4}"));
        metadata.insert("word_count".to_string(), word_count.to_string());

        let (message, suggestion, immediate_action) = if silence_ratio > 0.9 {
            (
                format!(
                    "Offline check for \"{text}\": audio is mostly silence \
                     ({:.0}% below the noise floor).",
                    silence_ratio * 100.0
                ),
                Some("Check your microphone input level and try again.".to_string()),
                "Check microphone input level".to_string(),
            )
        } else {
            (
                format!(
                    "Offline signal check for \"{text}\": {sample_count} samples, \
                     mean amplitude {mean_abs:.0}/{}.",
                    i16::MAX
                ),
                Some("Full pronunciation feedback requires an online connection.".to_string()),
                "Continue practicing".to_string(),
            )
        };

        let feedback_response = FeedbackResponse {
            feedback_items: vec![crate::traits::UserFeedback {
                message,
                suggestion,
                confidence,
                score: overall_score,
                priority: 0.5,
                metadata,
            }],
            overall_score,
            immediate_actions: vec![immediate_action],
            long_term_goals: vec!["Improve pronunciation accuracy".to_string()],
            progress_indicators: crate::traits::ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: if silence_ratio > 0.5 {
                    vec!["audio_input_level".to_string()]
                } else {
                    Vec::new()
                },
                stable_areas: Vec::new(),
                overall_trend: 0.0,
                completion_percentage: overall_score * 100.0,
            },
            processing_time: start.elapsed(),
            timestamp: chrono::Utc::now(),
            feedback_type: crate::FeedbackType::Quality,
        };

        Ok(feedback_response)
    }

    /// Save user progress offline
    pub fn save_progress_offline(&mut self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Save to local storage
        self.storage.save_user_progress(progress)?;

        // Queue for sync when online
        let operation = QueuedOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: OperationType::SaveProgress,
            session_id: progress.user_id.clone(),
            data: serde_json::to_value(progress).map_err(|e| OfflineError::SerializationError {
                message: format!("failed to serialize user progress for sync queue: {e}"),
            })?,
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        self.queue.add_operation(operation);

        Ok(())
    }

    /// Load user progress from offline storage
    pub fn load_progress_offline(
        &self,
        user_id: &str,
    ) -> Result<Option<UserProgress>, OfflineError> {
        self.storage.load_user_progress(user_id)
    }

    /// Sync offline operations when back online. Returns the real number of
    /// operations that were genuinely confirmed synced vs. those that were
    /// not (still queued for retry, or permanently dropped after
    /// exhausting retries) -- see [`SyncReport`]. Never silently discards
    /// data while reporting blanket success.
    async fn sync_offline_operations(&mut self) -> Result<SyncReport, OfflineError> {
        let operations = self.queue.get_pending_operations();
        let mut report = SyncReport::default();

        for operation in operations {
            match self.sync_operation(&operation).await {
                Ok(()) => {
                    self.queue.mark_completed(&operation.id);
                    report.synced += 1;
                }
                Err(e) => {
                    // Retry logic
                    if operation.retry_count < self.config.max_retries {
                        self.queue.increment_retry(&operation.id);
                    } else {
                        self.queue.mark_failed(&operation.id);
                    }
                    report.failed += 1;
                    eprintln!("Failed to sync operation {}: {}", operation.id, e);
                }
            }
        }

        Ok(report)
    }

    /// Sync an individual queued operation to the configured remote server
    /// via a real HTTP POST. Requires both the `microservices` feature
    /// (which provides the `reqwest` HTTP client) and a configured
    /// `config.sync_endpoint`; without either, this fails closed rather
    /// than reporting a fabricated success.
    #[cfg(feature = "microservices")]
    async fn sync_operation(&self, operation: &QueuedOperation) -> Result<(), OfflineError> {
        let Some(base_url) = self.config.sync_endpoint.as_ref() else {
            return Err(OfflineError::ConfigError {
                message: "cannot sync offline operations: no sync_endpoint configured on \
                          OfflineConfig"
                    .to_string(),
            });
        };

        // Install the pure-Rust rustls CryptoProvider before any TLS
        // handshake (reqwest is built with `rustls-no-provider`).
        // Once-guarded; safe to repeat.
        voirs_sdk::ensure_crypto_provider();

        let path = match operation.operation_type {
            OperationType::ProcessFeedback => "offline-sync/feedback",
            OperationType::SaveProgress => "offline-sync/progress",
            OperationType::SaveSession => "offline-sync/sessions",
        };
        let url = format!("{}/{path}", base_url.trim_end_matches('/'));

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.config.sync_timeout_secs))
            .build()
            .map_err(|e| OfflineError::NetworkError {
                message: format!("failed to build HTTP client for offline sync: {e}"),
            })?;

        let response = client
            .post(&url)
            .json(&operation.data)
            .send()
            .await
            .map_err(|e| OfflineError::NetworkError {
                message: format!("sync request to {url} failed: {e}"),
            })?;

        if !response.status().is_success() {
            return Err(OfflineError::NetworkError {
                message: format!("sync endpoint {url} returned HTTP {}", response.status()),
            });
        }

        Ok(())
    }

    /// Without the `microservices` feature, no real HTTP client is compiled
    /// in -- fail closed rather than fabricate a successful sync.
    #[cfg(not(feature = "microservices"))]
    async fn sync_operation(&self, _operation: &QueuedOperation) -> Result<(), OfflineError> {
        Err(OfflineError::ConfigError {
            message: "cannot sync offline operations: built without the 'microservices' \
                      feature, so no HTTP client is available"
                .to_string(),
        })
    }

    /// Get offline status
    #[must_use]
    pub fn get_offline_status(&self) -> OfflineStatus {
        OfflineStatus {
            is_offline: self.is_offline(),
            cached_models: self.cache.has_cached_models(),
            pending_operations: self.queue.get_pending_count(),
            storage_usage: self.storage.get_storage_usage(),
            last_sync: self.queue.get_last_sync_time(),
        }
    }

    /// Clear offline data
    pub fn clear_offline_data(&mut self) -> Result<(), OfflineError> {
        self.cache.clear_all()?;
        self.queue.clear_all();
        self.storage.clear_all()?;
        Ok(())
    }
}

/// Offline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineConfig {
    /// Enable offline mode
    pub enable_offline: bool,
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Cache expiration time in seconds
    pub cache_expiration_seconds: u64,
    /// Maximum number of queued operations
    pub max_queued_operations: u32,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Clear cache when going online
    pub clear_cache_on_online: bool,
    /// Storage directory for offline data. Also the root real model/data
    /// files are cached under (see `OfflineManager::new`).
    pub storage_directory: PathBuf,
    /// `"host:port"` endpoint used by [`OfflineManager::is_offline`]'s real
    /// connectivity probe.
    pub connectivity_check_endpoint: String,
    /// Timeout for a single connectivity probe attempt.
    pub connectivity_check_timeout_ms: u64,
    /// How long a connectivity probe result is cached before being
    /// re-checked.
    pub connectivity_cache_ttl_ms: u64,
    /// Base URL of the remote feedback-sync server, e.g.
    /// `"https://api.example.com"`. `None` (the default) means sync is not
    /// configured: queued operations fail closed instead of being silently
    /// marked as synced.
    pub sync_endpoint: Option<String>,
    /// HTTP request timeout for sync operations.
    pub sync_timeout_secs: u64,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            enable_offline: true,
            max_cache_size: 500 * 1024 * 1024,      // 500MB
            cache_expiration_seconds: 24 * 60 * 60, // 24 hours
            max_queued_operations: 1000,
            max_retries: 3,
            clear_cache_on_online: false,
            storage_directory: std::env::temp_dir().join("voirs_offline_data"),
            connectivity_check_endpoint: "1.1.1.1:443".to_string(),
            connectivity_check_timeout_ms: 500,
            connectivity_cache_ttl_ms: 5_000,
            sync_endpoint: None,
            sync_timeout_secs: 30,
        }
    }
}

/// Offline cache for storing models and data
pub struct OfflineCache {
    cached_models: HashMap<String, CachedModel>,
    cached_data: HashMap<String, CachedData>,
    total_size: u64,
    /// Real on-disk directory [`OfflineCache::preload_essential_data`] reads
    /// model/data files from. This crate does not bundle/embed any model
    /// data -- a real, non-empty file must actually exist here (see
    /// [`OfflineCache::with_models_dir`]) or preloading fails closed with
    /// [`OfflineError::ModelNotCached`] rather than fabricating placeholder
    /// bytes.
    models_dir: PathBuf,
}

impl Default for OfflineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl OfflineCache {
    /// Create a new offline cache that looks for real model/data files
    /// under `temp_dir()/voirs_offline_data/models`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_models_dir(
            std::env::temp_dir()
                .join("voirs_offline_data")
                .join("models"),
        )
    }

    /// Create a new offline cache that looks for real model/data files
    /// under `models_dir`.
    #[must_use]
    pub fn with_models_dir(models_dir: PathBuf) -> Self {
        Self {
            cached_models: HashMap::new(),
            cached_data: HashMap::new(),
            total_size: 0,
            models_dir,
        }
    }

    /// Check if essential models are cached
    #[must_use]
    pub fn has_cached_models(&self) -> bool {
        // Check if we have the essential models for offline operation
        self.cached_models.contains_key("pronunciation_model")
            && self.cached_models.contains_key("feedback_model")
    }

    /// Preload essential data for offline operation, reading each real
    /// model/data file from `models_dir` on disk. A missing, unreadable, or
    /// empty file fails closed with [`OfflineError::ModelNotCached`] -- no
    /// data is ever fabricated in its place.
    pub fn preload_essential_data(&mut self) -> Result<(), OfflineError> {
        // Preload pronunciation model
        let pronunciation_bytes = self.read_real_file("pronunciation_model.bin")?;
        self.cache_model("pronunciation_model", &pronunciation_bytes)?;

        // Preload feedback model
        let feedback_bytes = self.read_real_file("feedback_model.bin")?;
        self.cache_model("feedback_model", &feedback_bytes)?;

        // Preload common phrases
        let phrases_bytes = self.read_real_file("common_phrases.json")?;
        self.cache_data("common_phrases", &phrases_bytes)?;

        Ok(())
    }

    /// Read a real, non-empty file from `models_dir`. Fails closed with
    /// [`OfflineError::ModelNotCached`] (never fabricates bytes) if the
    /// file is missing, unreadable, or empty.
    fn read_real_file(&self, filename: &str) -> Result<Vec<u8>, OfflineError> {
        let path = self.models_dir.join(filename);
        let data = std::fs::read(&path).map_err(|_| OfflineError::ModelNotCached)?;
        if data.is_empty() {
            return Err(OfflineError::ModelNotCached);
        }
        Ok(data)
    }

    /// Cache a model
    fn cache_model(&mut self, model_id: &str, data: &[u8]) -> Result<(), OfflineError> {
        let model = CachedModel {
            id: model_id.to_string(),
            data: data.to_vec(),
            cached_at: chrono::Utc::now(),
            size: data.len() as u64,
        };

        self.total_size += model.size;
        self.cached_models.insert(model_id.to_string(), model);

        Ok(())
    }

    /// Cache data
    fn cache_data(&mut self, data_id: &str, data: &[u8]) -> Result<(), OfflineError> {
        let cached_data = CachedData {
            id: data_id.to_string(),
            data: data.to_vec(),
            cached_at: chrono::Utc::now(),
            size: data.len() as u64,
        };

        self.total_size += cached_data.size;
        self.cached_data.insert(data_id.to_string(), cached_data);

        Ok(())
    }

    /// Clear expired data
    pub fn clear_expired_data(&mut self) -> Result<(), OfflineError> {
        let now = chrono::Utc::now();
        let expiration_duration = chrono::Duration::hours(24);

        // Remove expired models
        self.cached_models.retain(|_, model| {
            let age = now.signed_duration_since(model.cached_at);
            if age > expiration_duration {
                self.total_size -= model.size;
                false
            } else {
                true
            }
        });

        // Remove expired data
        self.cached_data.retain(|_, data| {
            let age = now.signed_duration_since(data.cached_at);
            if age > expiration_duration {
                self.total_size -= data.size;
                false
            } else {
                true
            }
        });

        Ok(())
    }

    /// Clear all cached data
    pub fn clear_all(&mut self) -> Result<(), OfflineError> {
        self.cached_models.clear();
        self.cached_data.clear();
        self.total_size = 0;
        Ok(())
    }

    /// Get cache usage
    #[must_use]
    pub fn get_usage(&self) -> CacheUsage {
        CacheUsage {
            total_size: self.total_size,
            model_count: self.cached_models.len() as u32,
            data_count: self.cached_data.len() as u32,
        }
    }
}

/// Cached model
#[derive(Debug, Clone)]
struct CachedModel {
    id: String,
    data: Vec<u8>,
    cached_at: DateTime<Utc>,
    size: u64,
}

/// Cached data
#[derive(Debug, Clone)]
struct CachedData {
    id: String,
    data: Vec<u8>,
    cached_at: DateTime<Utc>,
    size: u64,
}

/// Operation queue for offline operations
pub struct OperationQueue {
    operations: HashMap<String, QueuedOperation>,
    last_sync: Option<DateTime<Utc>>,
}

impl Default for OperationQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationQueue {
    /// Create a new operation queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            last_sync: None,
        }
    }

    /// Add operation to queue
    pub fn add_operation(&mut self, operation: QueuedOperation) {
        self.operations.insert(operation.id.clone(), operation);
    }

    /// Get pending operations
    #[must_use]
    pub fn get_pending_operations(&self) -> Vec<QueuedOperation> {
        self.operations
            .values()
            .filter(|op| op.retry_count < 3)
            .cloned()
            .collect()
    }

    /// Mark operation as completed
    pub fn mark_completed(&mut self, operation_id: &str) {
        self.operations.remove(operation_id);
    }

    /// Mark operation as failed
    pub fn mark_failed(&mut self, operation_id: &str) {
        // In a real implementation, this would move to a failed operations list
        self.operations.remove(operation_id);
    }

    /// Increment retry count
    pub fn increment_retry(&mut self, operation_id: &str) {
        if let Some(operation) = self.operations.get_mut(operation_id) {
            operation.retry_count += 1;
        }
    }

    /// Get pending operations count
    #[must_use]
    pub fn get_pending_count(&self) -> u32 {
        self.operations.len() as u32
    }

    /// Get last sync time
    #[must_use]
    pub fn get_last_sync_time(&self) -> Option<DateTime<Utc>> {
        self.last_sync
    }

    /// Clear all operations
    pub fn clear_all(&mut self) {
        self.operations.clear();
    }
}

/// Queued operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedOperation {
    /// Description
    pub id: String,
    /// Description
    pub operation_type: OperationType,
    /// Description
    pub session_id: String,
    /// Description
    pub data: serde_json::Value,
    /// Description
    pub timestamp: DateTime<Utc>,
    /// Description
    pub retry_count: u32,
}

/// Operation type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Description
    ProcessFeedback,
    /// Description
    SaveProgress,
    /// Description
    SaveSession,
}

/// Offline storage
pub struct OfflineStorage {
    storage_directory: PathBuf,
    user_progress: HashMap<String, UserProgress>,
}

impl Default for OfflineStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl OfflineStorage {
    /// Create a new offline storage
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage_directory: std::env::temp_dir().join("voirs_offline_data"),
            user_progress: HashMap::new(),
        }
    }

    /// Create a new offline storage with custom directory
    #[must_use]
    pub fn with_directory(storage_directory: PathBuf) -> Self {
        Self {
            storage_directory,
            user_progress: HashMap::new(),
        }
    }

    /// Prepare offline storage
    pub fn prepare_offline_storage(&self) -> Result<(), OfflineError> {
        // Check if the path exists as a file
        if self.storage_directory.is_file() {
            return Err(OfflineError::StorageError {
                message: format!(
                    "Storage path exists as a file: {:?}",
                    self.storage_directory
                ),
            });
        }

        // Create storage directory if it doesn't exist
        if !self.storage_directory.exists() {
            std::fs::create_dir_all(&self.storage_directory).map_err(|e| {
                OfflineError::StorageError {
                    message: format!("Failed to create storage directory: {e}"),
                }
            })?;
        }

        Ok(())
    }

    /// Save user progress
    pub fn save_user_progress(&mut self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Store in memory first
        self.user_progress
            .insert(progress.user_id.clone(), progress.clone());

        // Try to persist to disk (optional, graceful fallback)
        if let Err(e) = self.try_persist_to_disk(progress) {
            // Log the error but don't fail the operation
            eprintln!("Warning: Could not persist to disk: {e}");
        }

        Ok(())
    }

    /// Try to persist user progress to disk (fallback method)
    fn try_persist_to_disk(&self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Ensure storage directory exists
        std::fs::create_dir_all(&self.storage_directory).map_err(|e| {
            OfflineError::StorageError {
                message: format!("Failed to create storage directory: {e}"),
            }
        })?;

        // Create safe filename (replace problematic characters)
        let safe_user_id = progress
            .user_id
            .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let file_path = self.storage_directory.join(format!("{safe_user_id}.json"));

        let json_data = serde_json::to_string_pretty(progress).map_err(|e| {
            OfflineError::SerializationError {
                message: format!("Failed to serialize user progress: {e}"),
            }
        })?;

        std::fs::write(&file_path, json_data).map_err(|e| OfflineError::StorageError {
            message: format!("Failed to write user progress: {e}"),
        })?;

        Ok(())
    }

    /// Load user progress
    pub fn load_user_progress(&self, user_id: &str) -> Result<Option<UserProgress>, OfflineError> {
        // First check in-memory cache
        if let Some(progress) = self.user_progress.get(user_id) {
            return Ok(Some(progress.clone()));
        }

        // Then check persistent storage
        let file_path = self.storage_directory.join(format!("{user_id}.json"));
        if file_path.exists() {
            let json_data =
                std::fs::read_to_string(&file_path).map_err(|e| OfflineError::StorageError {
                    message: format!("Failed to read user progress: {e}"),
                })?;

            let progress: UserProgress =
                serde_json::from_str(&json_data).map_err(|e| OfflineError::SerializationError {
                    message: format!("Failed to deserialize user progress: {e}"),
                })?;

            Ok(Some(progress))
        } else {
            Ok(None)
        }
    }

    /// Get storage usage: `used_bytes` is a real, recursive sum of file
    /// sizes under the storage directory; `available_bytes` is the real
    /// free space on that filesystem (via `df` on Linux/macOS; an honest
    /// `0` on platforms with no query path implemented here). Neither is a
    /// hardcoded placeholder.
    #[must_use]
    pub fn get_storage_usage(&self) -> StorageUsage {
        StorageUsage {
            used_bytes: compute_directory_size(&self.storage_directory),
            available_bytes: query_available_disk_bytes(&self.storage_directory),
        }
    }

    /// Clear all data
    pub fn clear_all(&mut self) -> Result<(), OfflineError> {
        self.user_progress.clear();

        // Really clear persistent storage: remove every real file this
        // instance wrote under `storage_directory` (not just the in-memory
        // cache above).
        if self.storage_directory.exists() {
            // Remove all files in the directory
            if let Ok(entries) = std::fs::read_dir(&self.storage_directory) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }

            // Try to remove the directory, but don't error if it fails
            let _ = std::fs::remove_dir(&self.storage_directory);
        }

        Ok(())
    }
}

/// Recursively sum real file sizes under `dir` (used for
/// [`OfflineStorage::get_storage_usage`]'s `used_bytes`). Symlinks are
/// skipped rather than followed (checked via `file_type()`, not
/// `metadata()`, which would implicitly follow a symlink) so a symlinked
/// subdirectory can never cause unbounded/infinite recursion. Honestly
/// returns `0` for a directory that does not exist yet (nothing has been
/// written there) rather than treating it as an error.
fn compute_directory_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };

    let mut total = 0u64;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            total = total.saturating_add(compute_directory_size(&entry.path()));
        } else if file_type.is_file() {
            if let Ok(metadata) = entry.metadata() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

/// Walk up from `path` to the nearest ancestor (inclusive) that actually
/// exists on disk. Used so a disk-space query still works before
/// `storage_directory` itself has been created (e.g. right after
/// `OfflineManager::new`, before `switch_to_offline` calls
/// `prepare_offline_storage`).
fn existing_ancestor(path: &Path) -> Option<&Path> {
    let mut candidate = path;
    loop {
        if candidate.exists() {
            return Some(candidate);
        }
        candidate = candidate.parent()?;
    }
}

/// Real available disk space (bytes) for the filesystem containing `path`,
/// queried via the platform's `df` utility (`-P` for POSIX single-line
/// output, so the "Available" column stays at a fixed index regardless of
/// how long the device name is). Honest `0` (never a fabricated constant)
/// if `df` is unavailable, fails, or on platforms with no query path
/// implemented here.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn query_available_disk_bytes(path: &Path) -> u64 {
    let Some(queryable_path) = existing_ancestor(path) else {
        return 0;
    };

    let Ok(output) = std::process::Command::new("df")
        .arg("-P")
        .arg("-k")
        .arg(queryable_path)
        .output()
    else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }
    let Ok(text) = String::from_utf8(output.stdout) else {
        return 0;
    };
    // POSIX `df -P` output: a header line, then one data line with columns
    // "Filesystem 1024-blocks Used Available Capacity Mounted-on".
    text.lines()
        .nth(1)
        .and_then(|line| line.split_whitespace().nth(3))
        .and_then(|s| s.parse::<u64>().ok())
        .map(|kb| kb.saturating_mul(1024))
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn query_available_disk_bytes(_path: &Path) -> u64 {
    0
}

/// Offline status
#[derive(Debug, Clone)]
pub struct OfflineStatus {
    /// Description
    pub is_offline: bool,
    /// Description
    pub cached_models: bool,
    /// Description
    pub pending_operations: u32,
    /// Description
    pub storage_usage: StorageUsage,
    /// Description
    pub last_sync: Option<DateTime<Utc>>,
}

/// Cache usage information
#[derive(Debug, Clone)]
pub struct CacheUsage {
    /// Description
    pub total_size: u64,
    /// Description
    pub model_count: u32,
    /// Description
    pub data_count: u32,
}

/// Storage usage information
#[derive(Debug, Clone)]
pub struct StorageUsage {
    /// Description
    pub used_bytes: u64,
    /// Description
    pub available_bytes: u64,
}

/// Offline error types
#[derive(Debug, thiserror::Error)]
pub enum OfflineError {
    #[error("Model not cached: Required model not available offline")]
    /// Description
    ModelNotCached,

    #[error("Storage error: {message}")]
    /// Description
    /// Description
    StorageError {
        /// Human-readable description of the storage issue.
        message: String,
    },

    #[error("Serialization error: {message}")]
    /// Description
    /// Description
    SerializationError {
        /// Human-readable description of the serialization issue.
        message: String,
    },

    #[error("Cache error: {message}")]
    /// Description
    /// Description
    CacheError {
        /// Human-readable description of the cache issue.
        message: String,
    },

    #[error("Network error: {message}")]
    /// Description
    /// Description
    NetworkError {
        /// Human-readable description of the network issue.
        message: String,
    },

    #[error("Configuration error: {message}")]
    /// Description
    /// Description
    ConfigError {
        /// Human-readable description of the configuration issue.
        message: String,
    },

    #[error("Invalid input: {message}")]
    /// Raised when a caller-supplied argument cannot be processed (e.g.
    /// empty audio data passed to offline feedback generation).
    InvalidInput {
        /// Human-readable description of what was invalid.
        message: String,
    },
}

/// Offline result type
pub type OfflineResult<T> = Result<T, OfflineError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Write real, non-empty fixture model/data files to `models_dir` --
    /// exactly what a real (if trivially small) on-disk model cache looks
    /// like from [`OfflineCache::preload_essential_data`]'s point of view.
    /// Tests seed these explicitly instead of relying on any bundled data,
    /// since this crate ships none.
    fn seed_real_model_files(models_dir: &std::path::Path) {
        std::fs::create_dir_all(models_dir).unwrap();
        std::fs::write(
            models_dir.join("pronunciation_model.bin"),
            b"real pronunciation model bytes",
        )
        .unwrap();
        std::fs::write(
            models_dir.join("feedback_model.bin"),
            b"real feedback model bytes",
        )
        .unwrap();
        std::fs::write(
            models_dir.join("common_phrases.json"),
            b"[\"hello\", \"world\"]",
        )
        .unwrap();
    }

    /// A fresh, unique temp directory for a test, so parallel tests never
    /// collide on the same on-disk storage/model directory.
    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("voirs_offline_{label}_{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn test_offline_config_default() {
        let config = OfflineConfig::default();
        assert!(config.enable_offline);
        assert_eq!(config.max_cache_size, 500 * 1024 * 1024);
        assert_eq!(config.cache_expiration_seconds, 24 * 60 * 60);
        assert_eq!(config.max_queued_operations, 1000);
        assert_eq!(config.max_retries, 3);
        assert!(!config.clear_cache_on_online);
        // Sync must be opt-in: no endpoint configured by default means
        // queued operations fail closed instead of a phantom server
        // silently "accepting" them.
        assert!(config.sync_endpoint.is_none());
        assert!(!config.connectivity_check_endpoint.is_empty());
    }

    #[test]
    fn test_offline_manager_creation() {
        let config = OfflineConfig::default();
        let manager = OfflineManager::new(config);
        // Must not panic; the real value depends on this sandbox's network
        // egress, so only coherence (not a fixed expectation) is checked
        // here -- see the dedicated connectivity tests below for
        // deterministic true/false cases.
        let _ = manager.is_offline();
    }

    /// `is_offline` must be backed by a real connectivity probe: a live
    /// local listener is genuinely reachable and must be reported online.
    #[test]
    fn test_is_offline_false_for_reachable_local_listener() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Keep something listening for the duration of the probe.
        std::thread::spawn(move || {
            let _ = listener.accept();
        });

        let config = OfflineConfig {
            connectivity_check_endpoint: addr.to_string(),
            connectivity_check_timeout_ms: 500,
            ..OfflineConfig::default()
        };
        let manager = OfflineManager::new(config);
        assert!(
            !manager.is_offline(),
            "a real, live local listener must be detected as online"
        );
    }

    /// A refused connection (nothing listening) must be reported offline --
    /// not a hardcoded `false` regardless of reality.
    #[test]
    fn test_is_offline_true_for_refused_local_port() {
        // Bind then immediately drop, freeing the port with nothing
        // listening on it -- a subsequent connect attempt gets a fast, real
        // refusal.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };

        let config = OfflineConfig {
            connectivity_check_endpoint: format!("127.0.0.1:{port}"),
            connectivity_check_timeout_ms: 300,
            ..OfflineConfig::default()
        };
        let manager = OfflineManager::new(config);
        assert!(
            manager.is_offline(),
            "a refused connection must be detected as offline"
        );
    }

    /// A cached, still-fresh probe result must be returned instead of
    /// re-probing -- verified by seeding a cache entry that contradicts
    /// what a fresh probe against the configured (unreachable) endpoint
    /// would report.
    #[test]
    fn test_is_offline_uses_cached_result_within_ttl() {
        let config = OfflineConfig {
            // Reserved-for-documentation address: never accepts real
            // connections, so if the cache were *not* consulted, a fresh
            // probe would report offline (`is_offline() == true`).
            connectivity_check_endpoint: "192.0.2.1:9".to_string(),
            connectivity_check_timeout_ms: 200,
            connectivity_cache_ttl_ms: 60_000,
            ..OfflineConfig::default()
        };
        let manager = OfflineManager::new(config);

        // Seed a fresh "online" cache entry directly, simulating "a real
        // probe just ran a moment ago and found us online".
        *manager
            .connectivity_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((Instant::now(), true));

        assert!(
            !manager.is_offline(),
            "a fresh cache entry within the TTL must be used instead of re-probing"
        );
    }

    #[test]
    fn test_offline_cache_operations() {
        let models_dir = unique_temp_dir("cache_ops");
        seed_real_model_files(&models_dir);
        let mut cache = OfflineCache::with_models_dir(models_dir.clone());

        // Initially no models cached
        assert!(!cache.has_cached_models());

        // Preload essential data from the real fixture files written above.
        cache.preload_essential_data().unwrap();

        // Should have cached models now
        assert!(cache.has_cached_models());

        // Check cache usage
        let usage = cache.get_usage();
        assert!(usage.total_size > 0);
        assert!(usage.model_count > 0);
        assert!(usage.data_count > 0);

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    /// Without any real model files on disk, preloading must fail closed
    /// (never fabricate placeholder bytes in their place).
    #[test]
    fn test_offline_cache_preload_fails_closed_without_real_files() {
        let models_dir = unique_temp_dir("cache_missing");
        // Deliberately do not create the directory or any files.
        let mut cache = OfflineCache::with_models_dir(models_dir);

        let result = cache.preload_essential_data();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OfflineError::ModelNotCached));
        assert!(!cache.has_cached_models());
    }

    #[test]
    fn test_operation_queue() {
        let mut queue = OperationQueue::new();

        // Initially empty
        assert_eq!(queue.get_pending_count(), 0);

        // Add operation
        let operation = QueuedOperation {
            id: "test_op".to_string(),
            operation_type: OperationType::ProcessFeedback,
            session_id: "session_1".to_string(),
            data: serde_json::json!({"test": "data"}),
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        queue.add_operation(operation);
        assert_eq!(queue.get_pending_count(), 1);

        // Mark as completed
        queue.mark_completed("test_op");
        assert_eq!(queue.get_pending_count(), 0);
    }

    #[test]
    fn test_offline_storage() {
        let mut storage = OfflineStorage::new();

        // Prepare storage directory
        storage.prepare_offline_storage().unwrap();

        // Test user progress operations
        let progress = UserProgress {
            user_id: "test_user".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: crate::traits::TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 5,
            total_practice_time: Duration::from_secs(3600),
        };

        // Save progress
        storage.save_user_progress(&progress).unwrap();

        // Load progress
        let loaded = storage.load_user_progress("test_user").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().user_id, "test_user");

        // Non-existent user
        let non_existent = storage.load_user_progress("non_existent").unwrap();
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_queued_operation_serialization() {
        let operation = QueuedOperation {
            id: "test_id".to_string(),
            operation_type: OperationType::SaveProgress,
            session_id: "session_123".to_string(),
            data: serde_json::json!({"progress": 85}),
            timestamp: chrono::Utc::now(),
            retry_count: 2,
        };

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: QueuedOperation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(operation.id, deserialized.id);
        assert_eq!(operation.operation_type, deserialized.operation_type);
        assert_eq!(operation.session_id, deserialized.session_id);
        assert_eq!(operation.data, deserialized.data);
        assert_eq!(operation.retry_count, deserialized.retry_count);
    }

    #[test]
    fn test_operation_type_equality() {
        assert_eq!(
            OperationType::ProcessFeedback,
            OperationType::ProcessFeedback
        );
        assert_eq!(OperationType::SaveProgress, OperationType::SaveProgress);
        assert_eq!(OperationType::SaveSession, OperationType::SaveSession);
        assert_ne!(OperationType::ProcessFeedback, OperationType::SaveProgress);
    }

    /// Build a `SessionState` fixture shared by several tests below.
    fn test_session() -> SessionState {
        SessionState {
            user_id: "test_user".to_string(),
            session_id: uuid::Uuid::new_v4(),
            start_time: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            current_task: None,
            stats: crate::traits::SessionStats::default(),
            preferences: crate::traits::UserPreferences::default(),
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics::default(),
        }
    }

    /// Encode a sequence of samples as little-endian 16-bit PCM bytes, the
    /// format `generate_offline_feedback` interprets `audio_data` as.
    fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[tokio::test]
    async fn test_offline_manager_operations() {
        let storage_dir = unique_temp_dir("manager_ops");
        seed_real_model_files(&storage_dir.join("models"));

        let config = OfflineConfig {
            storage_directory: storage_dir.clone(),
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);

        // Test switch to offline
        manager.switch_to_offline().unwrap();

        // Test offline status
        let status = manager.get_offline_status();
        assert!(status.cached_models);
        assert_eq!(status.pending_operations, 0);

        // Test offline feedback processing with real, non-trivial audio
        // (a mix of loud and near-silent samples) so the signal-derived
        // heuristic has real content to measure.
        let session = test_session();
        let samples: Vec<i16> = (0..4000)
            .map(|i| if i % 2 == 0 { 12_000 } else { 0 })
            .collect();
        let audio = pcm_bytes(&samples);

        let feedback = manager
            .process_feedback_offline(&session, &audio, "test text")
            .await;
        assert!(feedback.is_ok());

        let feedback_response = feedback.unwrap();
        assert!(!feedback_response.feedback_items.is_empty());
        // A real, in-range score derived from the input -- not necessarily
        // the old hardcoded 0.75.
        assert!((0.0..=1.0).contains(&feedback_response.overall_score));
        let item = &feedback_response.feedback_items[0];
        assert_eq!(
            item.metadata.get("sample_count").map(String::as_str),
            Some("4000")
        );

        // Should have queued operation
        let status = manager.get_offline_status();
        assert_eq!(status.pending_operations, 1);

        let _ = std::fs::remove_dir_all(&storage_dir);
    }

    /// `generate_offline_feedback` must actually use `audio_data` -- two
    /// different real inputs (loud vs. near-silent) must produce different
    /// real scores, not the same hardcoded constant either way.
    #[tokio::test]
    async fn test_offline_feedback_score_varies_with_real_audio() {
        let storage_dir = unique_temp_dir("feedback_varies");
        seed_real_model_files(&storage_dir.join("models"));
        let config = OfflineConfig {
            storage_directory: storage_dir.clone(),
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);
        manager.switch_to_offline().unwrap();

        let session = test_session();

        let loud_samples = vec![20_000i16; 4000];
        let loud_audio = pcm_bytes(&loud_samples);
        let loud_feedback = manager
            .process_feedback_offline(&session, &loud_audio, "hello world")
            .await
            .unwrap();

        let silent_samples = vec![0i16; 4000];
        let silent_audio = pcm_bytes(&silent_samples);
        let silent_feedback = manager
            .process_feedback_offline(&session, &silent_audio, "hello world")
            .await
            .unwrap();

        assert!(
            loud_feedback.overall_score > silent_feedback.overall_score,
            "louder real audio must score higher than near-silent real audio \
             (loud={}, silent={})",
            loud_feedback.overall_score,
            silent_feedback.overall_score
        );
        assert_ne!(
            loud_feedback.feedback_items[0].message, silent_feedback.feedback_items[0].message,
            "the message must reflect the real signal, not a constant"
        );

        let _ = std::fs::remove_dir_all(&storage_dir);
    }

    /// Empty audio data is invalid input, not something to fabricate a
    /// plausible-looking score for.
    #[tokio::test]
    async fn test_offline_feedback_rejects_empty_audio() {
        let storage_dir = unique_temp_dir("feedback_empty");
        seed_real_model_files(&storage_dir.join("models"));
        let config = OfflineConfig {
            storage_directory: storage_dir.clone(),
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);
        manager.switch_to_offline().unwrap();

        let session = test_session();
        let result = manager
            .process_feedback_offline(&session, &[], "text")
            .await;
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&storage_dir);
    }

    #[test]
    fn test_cache_cleanup() {
        let models_dir = unique_temp_dir("cache_cleanup");
        seed_real_model_files(&models_dir);
        let mut cache = OfflineCache::with_models_dir(models_dir.clone());

        // Add some data
        cache.preload_essential_data().unwrap();
        let initial_usage = cache.get_usage();

        // Clear expired data (in test, nothing should be expired)
        cache.clear_expired_data().unwrap();
        let usage_after_cleanup = cache.get_usage();
        assert_eq!(initial_usage.total_size, usage_after_cleanup.total_size);

        // Clear all data
        cache.clear_all().unwrap();
        let usage_after_clear = cache.get_usage();
        assert_eq!(usage_after_clear.total_size, 0);
        assert_eq!(usage_after_clear.model_count, 0);
        assert_eq!(usage_after_clear.data_count, 0);

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    /// `used_bytes` must be a real, recursive sum of the storage
    /// directory's actual file contents -- verified by writing a
    /// known-size file and checking the reported usage reflects it exactly,
    /// not a hardcoded `0`.
    #[test]
    fn test_storage_usage_reflects_real_files() {
        let dir = unique_temp_dir("storage_usage");
        std::fs::create_dir_all(&dir).unwrap();
        let storage = OfflineStorage::with_directory(dir.clone());

        let usage_empty = storage.get_storage_usage();
        assert_eq!(usage_empty.used_bytes, 0);

        std::fs::write(dir.join("data.bin"), vec![0u8; 12_345]).unwrap();
        let usage_with_file = storage.get_storage_usage();
        assert_eq!(
            usage_with_file.used_bytes, 12_345,
            "used_bytes must reflect the real file just written, not a hardcoded constant"
        );

        // A real subdirectory's contents must also be counted.
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("more.bin"), vec![0u8; 100]).unwrap();
        let usage_with_subdir = storage.get_storage_usage();
        assert_eq!(usage_with_subdir.used_bytes, 12_445);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_offline_progress_operations() {
        let temp_dir = unique_temp_dir("progress_ops");
        seed_real_model_files(&temp_dir.join("models"));

        let config = OfflineConfig {
            storage_directory: temp_dir.clone(),
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);

        // Switch to offline mode to prepare storage
        manager.switch_to_offline().unwrap();

        let progress = UserProgress {
            user_id: "test_user".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: crate::traits::TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: Duration::from_secs(7200),
        };

        // Save progress offline
        manager.save_progress_offline(&progress).unwrap();

        // Load progress offline
        let loaded = manager.load_progress_offline("test_user").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().user_id, "test_user");

        // Should have queued operation
        let status = manager.get_offline_status();
        assert_eq!(status.pending_operations, 1);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_online_offline_transitions() {
        let temp_dir = unique_temp_dir("transitions");
        seed_real_model_files(&temp_dir.join("models"));

        let config = OfflineConfig {
            storage_directory: temp_dir.clone(),
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);

        // Switch to offline
        manager.switch_to_offline().unwrap();
        let status = manager.get_offline_status();
        assert!(status.cached_models);

        // Switch back to online: nothing was queued, so this must succeed
        // even with no sync endpoint configured.
        manager.switch_to_online().await.unwrap();

        // Test clear data
        manager.clear_offline_data().unwrap();
        let status = manager.get_offline_status();
        assert!(!status.cached_models);
        assert_eq!(status.pending_operations, 0);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Without a configured `sync_endpoint`, `switch_to_online` must fail
    /// closed (real error) when operations are actually queued, rather than
    /// silently discarding them while reporting success.
    #[tokio::test]
    async fn test_switch_to_online_fails_closed_without_sync_endpoint() {
        let temp_dir = unique_temp_dir("no_sync_endpoint");
        seed_real_model_files(&temp_dir.join("models"));
        let config = OfflineConfig {
            storage_directory: temp_dir.clone(),
            ..OfflineConfig::default() // sync_endpoint: None
        };
        let mut manager = OfflineManager::new(config);
        manager.switch_to_offline().unwrap();

        manager
            .save_progress_offline(&UserProgress {
                user_id: "test_user".to_string(),
                ..UserProgress::default()
            })
            .unwrap();
        assert_eq!(manager.get_offline_status().pending_operations, 1);

        let result = manager.switch_to_online().await;
        assert!(
            result.is_err(),
            "switch_to_online must fail, not silently drop the queued operation"
        );
        // The operation must still be genuinely queued (retried), not
        // vanished as if it had been synced.
        assert!(manager.get_offline_status().pending_operations >= 1);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Drains a real client request from `socket` before a test responder
    /// replies. A single `read()` call is *not* guaranteed to return the
    /// whole request in one shot -- TCP is a byte stream, and the request
    /// (headers + JSON body) can legitimately arrive in more than one
    /// segment, especially under the scheduling jitter of a parallel test
    /// run. Replying (and letting the socket close via `Connection:
    /// close`) before the client has finished writing its body would race
    /// reqwest's write against our close and could surface as a spurious
    /// broken-pipe failure that has nothing to do with the behavior under
    /// test. This loops until it has seen the header terminator and
    /// (per any real `Content-Length`) the full declared body.
    #[cfg(feature = "microservices")]
    async fn read_full_http_request(socket: &mut tokio::net::TcpStream) {
        use tokio::io::AsyncReadExt;

        let mut received = Vec::new();
        let mut chunk = [0u8; 4096];
        while let Ok(n) = socket.read(&mut chunk).await {
            if n == 0 {
                break; // peer closed before/without completing the request
            }
            received.extend_from_slice(&chunk[..n]);

            let Some(header_end) = received
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|i| i + 4)
            else {
                continue; // headers not fully received yet
            };
            let headers = String::from_utf8_lossy(&received[..header_end]);
            let content_length: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if received.len() - header_end >= content_length {
                break; // full body (if any) has arrived
            }
        }
    }

    /// A real HTTP POST must actually reach a real local server and, on a
    /// real 200 response, the operation must be genuinely removed from the
    /// queue as synced.
    #[cfg(feature = "microservices")]
    #[tokio::test]
    async fn test_sync_operation_posts_to_real_local_server_and_succeeds() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            if let Ok((mut socket, _)) = listener.accept().await {
                read_full_http_request(&mut socket).await;
                let _ = socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
            }
        });

        let temp_dir = unique_temp_dir("sync_success");
        seed_real_model_files(&temp_dir.join("models"));
        let config = OfflineConfig {
            storage_directory: temp_dir.clone(),
            sync_endpoint: Some(format!("http://{addr}")),
            sync_timeout_secs: 5,
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);
        manager.switch_to_offline().unwrap();

        manager
            .save_progress_offline(&UserProgress {
                user_id: "test_user".to_string(),
                ..UserProgress::default()
            })
            .unwrap();
        assert_eq!(manager.get_offline_status().pending_operations, 1);

        manager.switch_to_online().await.unwrap();
        assert_eq!(
            manager.get_offline_status().pending_operations,
            0,
            "a real, confirmed sync must remove the operation from the queue"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// A server that genuinely rejects the request (HTTP 500) must result
    /// in a real, propagated failure -- not a fabricated success.
    #[cfg(feature = "microservices")]
    #[tokio::test]
    async fn test_sync_operation_propagates_real_server_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            if let Ok((mut socket, _)) = listener.accept().await {
                read_full_http_request(&mut socket).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\
                          Connection: close\r\n\r\n",
                    )
                    .await;
            }
        });

        let temp_dir = unique_temp_dir("sync_failure");
        seed_real_model_files(&temp_dir.join("models"));
        let config = OfflineConfig {
            storage_directory: temp_dir.clone(),
            sync_endpoint: Some(format!("http://{addr}")),
            sync_timeout_secs: 5,
            ..OfflineConfig::default()
        };
        let mut manager = OfflineManager::new(config);
        manager.switch_to_offline().unwrap();

        manager
            .save_progress_offline(&UserProgress {
                user_id: "test_user".to_string(),
                ..UserProgress::default()
            })
            .unwrap();

        let result = manager.switch_to_online().await;
        assert!(
            result.is_err(),
            "a real HTTP 500 must not be reported as success"
        );
        assert_eq!(
            manager.get_offline_status().pending_operations,
            1,
            "a genuinely failed sync must leave the operation queued for retry"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
