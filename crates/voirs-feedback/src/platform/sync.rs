//! Cross-platform synchronization support
//!
//! This module provides cross-platform synchronization capabilities for `VoiRS` feedback system
//! including data synchronization, conflict resolution, and offline support.

use crate::traits::UserProgress;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::time::timeout;

/// Synchronization manager for cross-platform data sync
pub struct SyncManager {
    config: SyncConfig,
    local_storage: Arc<RwLock<DefaultLocalStorage>>,
    remote_storage: Arc<RwLock<DefaultRemoteStorage>>,
    conflict_resolver: ConflictResolver,
    is_syncing: Arc<std::sync::atomic::AtomicBool>,
}

impl SyncManager {
    /// Create a new sync manager
    #[must_use]
    pub fn new(config: SyncConfig) -> Self {
        Self {
            config,
            local_storage: Arc::new(RwLock::new(DefaultLocalStorage::new())),
            remote_storage: Arc::new(RwLock::new(DefaultRemoteStorage::new())),
            conflict_resolver: ConflictResolver::new(),
            is_syncing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Synchronize data between local and remote storage
    pub async fn sync_data(&mut self) -> Result<SyncResult, SyncError> {
        // Check if already syncing to prevent concurrent sync operations
        if self.is_syncing.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SyncError::ConcurrencyError {
                message: "Sync already in progress".to_string(),
            });
        }

        self.is_syncing
            .store(true, std::sync::atomic::Ordering::Release);
        let _sync_guard = SyncGuard::new(self.is_syncing.clone());

        let sync_start = std::time::Instant::now();
        let mut sync_result = SyncResult::default();

        // Get local changes with timeout
        let local_changes = {
            // First acquire the lock (synchronously with timeout on the lock acquisition)
            let local_storage_guard = timeout(
                Duration::from_secs(self.config.operation_timeout_seconds),
                async {
                    self.local_storage
                        .read()
                        .map_err(|_| SyncError::StorageError {
                            message: "Failed to acquire local storage read lock".to_string(),
                        })
                },
            )
            .await
            .map_err(|_| SyncError::StorageError {
                message: "Timeout acquiring local storage lock".to_string(),
            })??;

            // Clone the changes while holding the lock, then drop the lock
            let changes = local_storage_guard.changes.clone();
            drop(local_storage_guard); // Explicitly drop the lock

            // Now we can safely return without holding the lock
            Ok::<_, SyncError>(changes)
        }?;

        // Get remote changes with timeout
        let remote_changes = {
            // First acquire the lock (synchronously with timeout on the lock acquisition)
            let remote_storage_guard = timeout(
                Duration::from_secs(self.config.operation_timeout_seconds),
                async {
                    self.remote_storage
                        .read()
                        .map_err(|_| SyncError::StorageError {
                            message: "Failed to acquire remote storage read lock".to_string(),
                        })
                },
            )
            .await
            .map_err(|_| SyncError::StorageError {
                message: "Timeout acquiring remote storage lock".to_string(),
            })??;

            // Clone the changes while holding the lock, then drop the lock
            let changes = remote_storage_guard.changes.clone();
            drop(remote_storage_guard); // Explicitly drop the lock

            // Now we can safely return without holding the lock
            Ok::<_, SyncError>(changes)
        }?;

        // Resolve conflicts
        let resolved_changes = self
            .conflict_resolver
            .resolve_conflicts(local_changes, remote_changes)?;

        // Apply changes locally with timeout
        for change in &resolved_changes.local_changes {
            // Acquire the lock
            let mut local_storage_guard = timeout(
                Duration::from_secs(self.config.operation_timeout_seconds),
                async {
                    self.local_storage
                        .write()
                        .map_err(|_| SyncError::StorageError {
                            message: "Failed to acquire local storage write lock".to_string(),
                        })
                },
            )
            .await
            .map_err(|_| SyncError::StorageError {
                message: "Timeout acquiring local storage write lock".to_string(),
            })??;

            // Apply change synchronously (the method is async but doesn't actually await)
            local_storage_guard.changes.push(change.clone());
            drop(local_storage_guard); // Explicitly drop the lock
            sync_result.local_changes_applied += 1;
        }

        // Apply changes remotely with timeout
        for change in &resolved_changes.remote_changes {
            // Acquire the lock
            let mut remote_storage_guard = timeout(
                Duration::from_secs(self.config.operation_timeout_seconds),
                async {
                    self.remote_storage
                        .write()
                        .map_err(|_| SyncError::StorageError {
                            message: "Failed to acquire remote storage write lock".to_string(),
                        })
                },
            )
            .await
            .map_err(|_| SyncError::StorageError {
                message: "Timeout acquiring remote storage write lock".to_string(),
            })??;

            // Apply change synchronously (the method is async but doesn't actually await)
            remote_storage_guard.changes.push(change.clone());
            drop(remote_storage_guard); // Explicitly drop the lock
            sync_result.remote_changes_applied += 1;
        }

        sync_result.conflicts_resolved = resolved_changes.conflicts_resolved;
        sync_result.last_sync_time = chrono::Utc::now();
        sync_result.sync_duration = sync_start.elapsed();

        Ok(sync_result)
    }

    /// Check if sync is needed
    #[must_use]
    pub fn needs_sync(&self) -> bool {
        // Don't sync if already syncing
        if self.is_syncing.load(std::sync::atomic::Ordering::Acquire) {
            return false;
        }

        // Check if enough time has passed since last sync
        if let Some(last_sync) = self.get_last_sync_time() {
            let elapsed = chrono::Utc::now()
                .signed_duration_since(last_sync)
                .to_std()
                .unwrap_or(Duration::ZERO);
            elapsed >= Duration::from_secs(self.config.sync_interval_seconds)
        } else {
            true
        }
    }

    /// Get last sync time
    #[must_use]
    pub fn get_last_sync_time(&self) -> Option<DateTime<Utc>> {
        // This would be stored in local storage
        None
    }

    /// Force sync regardless of timing
    pub async fn force_sync(&mut self) -> Result<SyncResult, SyncError> {
        self.sync_data().await
    }

    /// Get sync status
    #[must_use]
    pub fn get_sync_status(&self) -> SyncStatus {
        let pending_changes = self.get_pending_changes_count();
        SyncStatus {
            is_syncing: self.is_syncing.load(std::sync::atomic::Ordering::Acquire),
            last_sync_time: self.get_last_sync_time(),
            pending_changes,
            network_available: self.check_network_availability(),
            sync_enabled: self.config.enabled,
        }
    }

    /// Get pending changes count
    fn get_pending_changes_count(&self) -> u32 {
        if let Ok(storage) = self.local_storage.try_read() {
            storage.changes.len() as u32
        } else {
            0
        }
    }

    /// Check network availability
    fn check_network_availability(&self) -> bool {
        // In a real implementation, this would check actual network connectivity
        true
    }
}

/// Synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Enable synchronization
    pub enabled: bool,
    /// Sync interval in seconds
    pub sync_interval_seconds: u64,
    /// Remote server URL
    pub remote_url: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable conflict resolution
    pub enable_conflict_resolution: bool,
    /// Sync only on `WiFi`
    pub wifi_only: bool,
    /// Operation timeout in seconds
    pub operation_timeout_seconds: u64,
    /// Maximum concurrent sync operations
    pub max_concurrent_operations: u32,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval_seconds: 300, // 5 minutes
            remote_url: "https://api.voirs.com/sync".to_string(),
            auth_token: None,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_conflict_resolution: true,
            wifi_only: false,
            operation_timeout_seconds: 30,
            max_concurrent_operations: 1,
        }
    }
}

/// Synchronization result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of local changes applied
    pub local_changes_applied: u32,
    /// Number of remote changes applied
    pub remote_changes_applied: u32,
    /// Number of conflicts resolved
    pub conflicts_resolved: u32,
    /// Last sync time
    pub last_sync_time: DateTime<Utc>,
    /// Sync duration
    pub sync_duration: Duration,
}

impl Default for SyncResult {
    fn default() -> Self {
        Self {
            local_changes_applied: 0,
            remote_changes_applied: 0,
            conflicts_resolved: 0,
            last_sync_time: chrono::Utc::now(),
            sync_duration: Duration::from_secs(0),
        }
    }
}

/// Synchronization status
#[derive(Debug, Clone)]
pub struct SyncStatus {
    /// Whether sync is currently running
    pub is_syncing: bool,
    /// Last sync time
    pub last_sync_time: Option<DateTime<Utc>>,
    /// Number of pending changes
    pub pending_changes: u32,
    /// Network availability
    pub network_available: bool,
    /// Sync enabled status
    pub sync_enabled: bool,
}

/// Data change for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChange {
    /// Unique change ID
    pub id: String,
    /// Change type
    pub change_type: ChangeType,
    /// Entity type
    pub entity_type: String,
    /// Entity ID
    pub entity_id: String,
    /// Change data
    pub data: serde_json::Value,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Device ID that made the change
    pub device_id: String,
}

/// Change type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Description
    Create,
    /// Description
    Update,
    /// Description
    Delete,
}

/// Conflict resolver
pub struct ConflictResolver {
    resolution_strategy: ConflictResolutionStrategy,
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictResolver {
    /// Create a new conflict resolver using the default `LastWriteWins` strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            resolution_strategy: ConflictResolutionStrategy::LastWriteWins,
        }
    }

    /// Create a conflict resolver using a specific resolution strategy (e.g.
    /// [`ConflictResolutionStrategy::Merge`] for field-level merging instead
    /// of whole-record last-write-wins).
    #[must_use]
    pub fn with_strategy(resolution_strategy: ConflictResolutionStrategy) -> Self {
        Self {
            resolution_strategy,
        }
    }

    /// Resolve conflicts between local and remote changes
    pub fn resolve_conflicts(
        &self,
        local_changes: Vec<DataChange>,
        remote_changes: Vec<DataChange>,
    ) -> Result<ResolvedChanges, SyncError> {
        let mut resolved = ResolvedChanges::default();
        let mut conflicts_by_entity: HashMap<String, Vec<DataChange>> = HashMap::new();

        // Group changes by entity
        for change in local_changes {
            conflicts_by_entity
                .entry(change.entity_id.clone())
                .or_default()
                .push(change);
        }

        for change in remote_changes {
            conflicts_by_entity
                .entry(change.entity_id.clone())
                .or_default()
                .push(change);
        }

        // Resolve conflicts for each entity
        for (entity_id, changes) in conflicts_by_entity {
            if changes.len() > 1 {
                // Conflict detected
                let resolved_change = self.resolve_entity_conflict(&entity_id, changes)?;
                resolved.local_changes.push(resolved_change.clone());
                resolved.remote_changes.push(resolved_change);
                resolved.conflicts_resolved += 1;
            } else if let Some(change) = changes.into_iter().next() {
                // No conflict, apply change
                if change.device_id == "local" {
                    resolved.remote_changes.push(change);
                } else {
                    resolved.local_changes.push(change);
                }
            }
        }

        Ok(resolved)
    }

    /// Resolve conflict for a specific entity
    fn resolve_entity_conflict(
        &self,
        entity_id: &str,
        changes: Vec<DataChange>,
    ) -> Result<DataChange, SyncError> {
        if changes.is_empty() {
            return Err(SyncError::ConflictError {
                message: format!("no changes to resolve for entity {entity_id}"),
            });
        }

        match self.resolution_strategy {
            ConflictResolutionStrategy::LastWriteWins => {
                // Use the change with the latest timestamp
                changes
                    .into_iter()
                    .max_by_key(|c| c.timestamp)
                    .ok_or_else(|| SyncError::ConflictError {
                        message: format!("no changes to resolve for entity {entity_id}"),
                    })
            }
            ConflictResolutionStrategy::FirstWriteWins => {
                // Use the change with the earliest timestamp
                changes
                    .into_iter()
                    .min_by_key(|c| c.timestamp)
                    .ok_or_else(|| SyncError::ConflictError {
                        message: format!("no changes to resolve for entity {entity_id}"),
                    })
            }
            ConflictResolutionStrategy::Merge => self.merge_changes(entity_id, changes),
        }
    }

    /// Merge a set of conflicting changes for a single entity at the JSON
    /// field level.
    ///
    /// Whole-record last-write-wins (as used by [`ConflictResolutionStrategy::LastWriteWins`])
    /// is too coarse when two devices edited *different* fields of the same
    /// entity concurrently: picking "the latest change" outright discards
    /// every field the losing change touched, even fields the winner never
    /// set. This merges per JSON field instead: for each key that appears in
    /// any change's `data` object, the value from whichever change carries
    /// the newest timestamp *for that key* wins, so edits to disjoint fields
    /// are all preserved and only genuinely conflicting fields fall back to
    /// last-write-wins.
    fn merge_changes(
        &self,
        entity_id: &str,
        mut changes: Vec<DataChange>,
    ) -> Result<DataChange, SyncError> {
        // Oldest first, so that when two changes touch the same field the
        // later loop iteration (by definition, the later timestamp) is the
        // one that ends up applied last.
        changes.sort_by_key(|c| c.timestamp);

        let latest = changes
            .last()
            .cloned()
            .ok_or_else(|| SyncError::ConflictError {
                message: format!("no changes to merge for entity {entity_id}"),
            })?;

        let mut merged_fields = serde_json::Map::new();
        let mut field_timestamps: HashMap<String, DateTime<Utc>> = HashMap::new();

        for change in &changes {
            let Some(fields) = change.data.as_object() else {
                continue;
            };

            for (key, value) in fields {
                let should_apply = field_timestamps
                    .get(key)
                    .is_none_or(|existing| change.timestamp >= *existing);

                if should_apply {
                    merged_fields.insert(key.clone(), value.clone());
                    field_timestamps.insert(key.clone(), change.timestamp);
                }
            }
        }

        Ok(DataChange {
            id: latest.id,
            change_type: latest.change_type,
            entity_type: latest.entity_type,
            entity_id: latest.entity_id,
            data: serde_json::Value::Object(merged_fields),
            timestamp: latest.timestamp,
            device_id: latest.device_id,
        })
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictResolutionStrategy {
    /// Use the change with the latest timestamp
    LastWriteWins,
    /// Use the change with the earliest timestamp
    FirstWriteWins,
    /// Merge changes together
    Merge,
}

/// Resolved changes
#[derive(Debug, Clone, Default)]
pub struct ResolvedChanges {
    /// Changes to apply locally
    pub local_changes: Vec<DataChange>,
    /// Changes to apply remotely
    pub remote_changes: Vec<DataChange>,
    /// Number of conflicts resolved
    pub conflicts_resolved: u32,
}

/// Local storage trait
#[async_trait]
pub trait LocalStorage: Send + Sync {
    /// Get pending changes
    async fn get_pending_changes(&self) -> Result<Vec<DataChange>, SyncError>;
    /// Apply a change
    async fn apply_change(&mut self, change: &DataChange) -> Result<(), SyncError>;
    /// Store user progress
    async fn store_user_progress(&mut self, progress: &UserProgress) -> Result<(), SyncError>;
    /// Get user progress
    async fn get_user_progress(&self, user_id: &str) -> Result<Option<UserProgress>, SyncError>;
}

/// Remote storage trait
#[async_trait]
pub trait RemoteStorage: Send + Sync {
    /// Get remote changes
    async fn get_remote_changes(&self) -> Result<Vec<DataChange>, SyncError>;
    /// Apply a change
    async fn apply_change(&mut self, change: &DataChange) -> Result<(), SyncError>;
    /// Upload user progress
    async fn upload_user_progress(&mut self, progress: &UserProgress) -> Result<(), SyncError>;
    /// Download user progress
    async fn download_user_progress(
        &self,
        user_id: &str,
    ) -> Result<Option<UserProgress>, SyncError>;
}

/// Default local storage implementation
pub struct DefaultLocalStorage {
    changes: Vec<DataChange>,
    user_progress: HashMap<String, UserProgress>,
}

impl Default for DefaultLocalStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultLocalStorage {
    /// Create a new default local storage
    #[must_use]
    pub fn new() -> Self {
        Self {
            changes: Vec::new(),
            user_progress: HashMap::new(),
        }
    }
}

#[async_trait]
impl LocalStorage for DefaultLocalStorage {
    async fn get_pending_changes(&self) -> Result<Vec<DataChange>, SyncError> {
        Ok(self.changes.clone())
    }

    async fn apply_change(&mut self, change: &DataChange) -> Result<(), SyncError> {
        self.changes.push(change.clone());
        Ok(())
    }

    async fn store_user_progress(&mut self, progress: &UserProgress) -> Result<(), SyncError> {
        self.user_progress
            .insert(progress.user_id.clone(), progress.clone());
        Ok(())
    }

    async fn get_user_progress(&self, user_id: &str) -> Result<Option<UserProgress>, SyncError> {
        Ok(self.user_progress.get(user_id).cloned())
    }
}

/// Default remote storage implementation
pub struct DefaultRemoteStorage {
    changes: Vec<DataChange>,
    user_progress: HashMap<String, UserProgress>,
}

impl Default for DefaultRemoteStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultRemoteStorage {
    /// Create a new default remote storage
    #[must_use]
    pub fn new() -> Self {
        Self {
            changes: Vec::new(),
            user_progress: HashMap::new(),
        }
    }
}

#[async_trait]
impl RemoteStorage for DefaultRemoteStorage {
    async fn get_remote_changes(&self) -> Result<Vec<DataChange>, SyncError> {
        Ok(self.changes.clone())
    }

    async fn apply_change(&mut self, change: &DataChange) -> Result<(), SyncError> {
        self.changes.push(change.clone());
        Ok(())
    }

    async fn upload_user_progress(&mut self, progress: &UserProgress) -> Result<(), SyncError> {
        self.user_progress
            .insert(progress.user_id.clone(), progress.clone());
        Ok(())
    }

    async fn download_user_progress(
        &self,
        user_id: &str,
    ) -> Result<Option<UserProgress>, SyncError> {
        Ok(self.user_progress.get(user_id).cloned())
    }
}

/// Synchronization error types
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error("Network error: {message}")]
    /// Description
    /// Description
    NetworkError {
        /// Human-readable description of the network issue.
        message: String,
    },

    #[error("Storage error: {message}")]
    /// Description
    /// Description
    StorageError {
        /// Human-readable description of the storage issue.
        message: String,
    },

    #[error("Conflict resolution error: {message}")]
    /// Description
    /// Description
    ConflictError {
        /// Details about the conflict that could not be resolved.
        message: String,
    },

    #[error("Authentication error: {message}")]
    /// Description
    /// Description
    AuthError {
        /// Human-readable description of the authentication failure.
        message: String,
    },

    #[error("Serialization error: {message}")]
    /// Description
    /// Description
    SerializationError {
        /// Human-readable description of the serialization issue.
        message: String,
    },

    #[error("Configuration error: {message}")]
    /// Description
    /// Description
    ConfigError {
        /// Human-readable description of the configuration issue.
        message: String,
    },

    #[error("Concurrency error: {message}")]
    /// Description
    /// Description
    ConcurrencyError {
        /// Human-readable description of the concurrency issue.
        message: String,
    },

    #[error("Timeout error: {message}")]
    /// Description
    /// Description
    TimeoutError {
        /// Human-readable description of the timeout condition.
        message: String,
    },
}

/// Synchronization result type
pub type SyncResult2<T> = Result<T, SyncError>;

/// RAII guard for sync operations
struct SyncGuard {
    is_syncing: Arc<std::sync::atomic::AtomicBool>,
}

impl SyncGuard {
    fn new(is_syncing: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self { is_syncing }
    }
}

impl Drop for SyncGuard {
    fn drop(&mut self) {
        self.is_syncing
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_config_default() {
        let config = SyncConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sync_interval_seconds, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_conflict_resolution);
        assert!(!config.wifi_only);
    }

    #[test]
    fn test_sync_manager_creation() {
        let config = SyncConfig::default();
        let manager = SyncManager::new(config);
        assert!(manager.needs_sync());
    }

    #[test]
    fn test_conflict_resolver_last_write_wins() {
        let resolver = ConflictResolver::new();

        let mut changes = vec![
            DataChange {
                id: "1".to_string(),
                change_type: ChangeType::Update,
                entity_type: "user_progress".to_string(),
                entity_id: "user_1".to_string(),
                data: serde_json::json!({"score": 80}),
                timestamp: chrono::Utc::now(),
                device_id: "device_1".to_string(),
            },
            DataChange {
                id: "2".to_string(),
                change_type: ChangeType::Update,
                entity_type: "user_progress".to_string(),
                entity_id: "user_1".to_string(),
                data: serde_json::json!({"score": 90}),
                timestamp: chrono::Utc::now() + chrono::Duration::seconds(1),
                device_id: "device_2".to_string(),
            },
        ];

        let resolved_change = resolver
            .resolve_entity_conflict("user_1", changes.clone())
            .unwrap();
        assert_eq!(resolved_change.id, "2");
        assert_eq!(resolved_change.data["score"], 90);
    }

    #[test]
    fn test_data_change_serialization() {
        let change = DataChange {
            id: "test_id".to_string(),
            change_type: ChangeType::Create,
            entity_type: "user_progress".to_string(),
            entity_id: "user_123".to_string(),
            data: serde_json::json!({"score": 85}),
            timestamp: chrono::Utc::now(),
            device_id: "test_device".to_string(),
        };

        let serialized = serde_json::to_string(&change).unwrap();
        let deserialized: DataChange = serde_json::from_str(&serialized).unwrap();

        assert_eq!(change.id, deserialized.id);
        assert_eq!(change.change_type, deserialized.change_type);
        assert_eq!(change.entity_type, deserialized.entity_type);
        assert_eq!(change.entity_id, deserialized.entity_id);
        assert_eq!(change.data, deserialized.data);
        assert_eq!(change.device_id, deserialized.device_id);
    }

    #[test]
    fn test_change_type_equality() {
        assert_eq!(ChangeType::Create, ChangeType::Create);
        assert_eq!(ChangeType::Update, ChangeType::Update);
        assert_eq!(ChangeType::Delete, ChangeType::Delete);
        assert_ne!(ChangeType::Create, ChangeType::Update);
    }

    #[test]
    fn test_conflict_resolution_strategy() {
        let strategy = ConflictResolutionStrategy::LastWriteWins;
        assert_eq!(strategy, ConflictResolutionStrategy::LastWriteWins);
        assert_ne!(strategy, ConflictResolutionStrategy::FirstWriteWins);
        assert_ne!(strategy, ConflictResolutionStrategy::Merge);
    }

    /// Real field-level merge: two devices concurrently edit *different*
    /// fields of the same entity. A whole-record last-write-wins strategy
    /// would silently drop the loser's field entirely; the Merge strategy
    /// must preserve both edits and, for a field touched by both changes,
    /// resolve it using the newer of the two per-field timestamps.
    #[test]
    fn test_merge_strategy_combines_disjoint_and_conflicting_fields() {
        let resolver = ConflictResolver::with_strategy(ConflictResolutionStrategy::Merge);

        let t0 = chrono::Utc::now();
        let changes = vec![
            DataChange {
                id: "device_a_change".to_string(),
                change_type: ChangeType::Update,
                entity_type: "user_progress".to_string(),
                entity_id: "user_1".to_string(),
                // Sets `score` (later overwritten by device B) and `streak`
                // (unique to this change; must survive the merge).
                data: serde_json::json!({"score": 80, "streak": 5}),
                timestamp: t0,
                device_id: "device_a".to_string(),
            },
            DataChange {
                id: "device_b_change".to_string(),
                change_type: ChangeType::Update,
                entity_type: "user_progress".to_string(),
                entity_id: "user_1".to_string(),
                // Sets `score` (should win, later timestamp) and `name`
                // (unique to this change; must survive the merge).
                data: serde_json::json!({"score": 95, "name": "Alice"}),
                timestamp: t0 + chrono::Duration::seconds(10),
                device_id: "device_b".to_string(),
            },
        ];

        let merged = resolver.resolve_entity_conflict("user_1", changes).unwrap();
        let fields = merged.data.as_object().expect("merged data is an object");

        // The later change's value wins for the field both changes touched.
        assert_eq!(fields["score"], serde_json::json!(95));
        // Fields unique to each change must both be preserved, not dropped.
        assert_eq!(fields["streak"], serde_json::json!(5));
        assert_eq!(fields["name"], serde_json::json!("Alice"));
        assert_eq!(fields.len(), 3, "no field should be lost or duplicated");
    }

    #[test]
    fn test_resolve_entity_conflict_empty_changes_is_an_honest_error() {
        let resolver = ConflictResolver::new();
        assert!(resolver.resolve_entity_conflict("user_1", vec![]).is_err());

        let merge_resolver = ConflictResolver::with_strategy(ConflictResolutionStrategy::Merge);
        assert!(merge_resolver
            .resolve_entity_conflict("user_1", vec![])
            .is_err());
    }

    #[tokio::test]
    async fn test_local_storage_operations() {
        let mut storage = DefaultLocalStorage::new();

        // Test storing and retrieving user progress
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

        storage.store_user_progress(&progress).await.unwrap();
        let retrieved = storage.get_user_progress("test_user").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().user_id, "test_user");

        // Test non-existent user
        let non_existent = storage.get_user_progress("non_existent").await.unwrap();
        assert!(non_existent.is_none());
    }

    #[tokio::test]
    async fn test_remote_storage_operations() {
        let mut storage = DefaultRemoteStorage::new();

        // Test storing and retrieving user progress
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

        storage.upload_user_progress(&progress).await.unwrap();
        let retrieved = storage.download_user_progress("test_user").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().user_id, "test_user");

        // Test non-existent user
        let non_existent = storage
            .download_user_progress("non_existent")
            .await
            .unwrap();
        assert!(non_existent.is_none());
    }

    #[tokio::test]
    async fn test_sync_manager_sync_data() {
        let config = SyncConfig::default();
        let mut manager = SyncManager::new(config);

        // Test sync operation
        let result = manager.sync_data().await;
        assert!(result.is_ok());

        let sync_result = result.unwrap();
        assert_eq!(sync_result.local_changes_applied, 0);
        assert_eq!(sync_result.remote_changes_applied, 0);
        assert_eq!(sync_result.conflicts_resolved, 0);
    }

    #[test]
    fn test_sync_status() {
        let config = SyncConfig::default();
        let manager = SyncManager::new(config);

        let status = manager.get_sync_status();
        assert!(!status.is_syncing);
        assert!(status.network_available);
        assert!(status.sync_enabled);
        assert_eq!(status.pending_changes, 0);
    }

    #[tokio::test]
    async fn test_conflict_resolution_with_multiple_changes() {
        let resolver = ConflictResolver::new();

        let local_changes = vec![DataChange {
            id: "local_1".to_string(),
            change_type: ChangeType::Update,
            entity_type: "user_progress".to_string(),
            entity_id: "user_1".to_string(),
            data: serde_json::json!({"score": 80}),
            timestamp: chrono::Utc::now(),
            device_id: "local".to_string(),
        }];

        let remote_changes = vec![DataChange {
            id: "remote_1".to_string(),
            change_type: ChangeType::Update,
            entity_type: "user_progress".to_string(),
            entity_id: "user_1".to_string(),
            data: serde_json::json!({"score": 85}),
            timestamp: chrono::Utc::now() + chrono::Duration::seconds(1),
            device_id: "remote".to_string(),
        }];

        let resolved = resolver
            .resolve_conflicts(local_changes, remote_changes)
            .unwrap();
        assert_eq!(resolved.conflicts_resolved, 1);
        assert_eq!(resolved.local_changes.len(), 1);
        assert_eq!(resolved.remote_changes.len(), 1);
    }
}
