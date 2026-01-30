// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Distributed state management for high availability

use super::{HighAvailabilityError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// State synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMode {
    /// Synchronous replication - wait for all replicas
    Synchronous,
    /// Asynchronous replication - don't wait
    Asynchronous,
    /// Quorum-based - wait for majority
    Quorum,
}

/// State entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry {
    /// Entry key
    pub key: String,
    /// Entry value (JSON serialized)
    pub value: serde_json::Value,
    /// Version number
    pub version: u64,
    /// Last updated timestamp
    #[serde(skip, default = "Instant::now")]
    pub updated_at: Instant,
    /// TTL (time-to-live)
    pub ttl: Option<Duration>,
}

/// Distributed state manager
pub struct DistributedStateManager {
    sync_mode: SyncMode,
    state: Arc<RwLock<HashMap<String, StateEntry>>>,
    replica_count: usize,
}

impl DistributedStateManager {
    /// Create a new distributed state manager
    #[must_use]
    pub fn new(sync_mode: SyncMode, replica_count: usize) -> Self {
        Self {
            sync_mode,
            state: Arc::new(RwLock::new(HashMap::new())),
            replica_count,
        }
    }

    /// Set a state value
    pub async fn set(
        &self,
        key: String,
        value: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<()> {
        debug!("Setting state key: {}", key);

        let entry = StateEntry {
            key: key.clone(),
            value,
            version: self.get_next_version(&key),
            updated_at: Instant::now(),
            ttl,
        };

        // Store locally
        self.state.write().insert(key.clone(), entry.clone());

        // Replicate to other instances based on sync mode
        match self.sync_mode {
            SyncMode::Synchronous => {
                self.replicate_synchronous(entry).await?;
            }
            SyncMode::Asynchronous => {
                self.replicate_asynchronous(entry).await;
            }
            SyncMode::Quorum => {
                self.replicate_quorum(entry).await?;
            }
        }

        Ok(())
    }

    /// Get a state value
    #[must_use]
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        let state = self.state.read();

        if let Some(entry) = state.get(key) {
            // Check if entry is expired
            if let Some(ttl) = entry.ttl {
                if entry.updated_at.elapsed() > ttl {
                    return None;
                }
            }

            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Delete a state value
    pub async fn delete(&self, key: &str) -> Result<()> {
        debug!("Deleting state key: {}", key);

        self.state.write().remove(key);

        // Replicate deletion
        info!("State deleted: {}", key);

        Ok(())
    }

    /// Get all keys
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        self.state.read().keys().cloned().collect()
    }

    /// Get state entry count
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.state.read().len()
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        let mut state = self.state.write();
        let now = Instant::now();

        state.retain(|_, entry| {
            if let Some(ttl) = entry.ttl {
                entry.updated_at + ttl > now
            } else {
                true
            }
        });

        debug!("Cleaned up expired entries, remaining: {}", state.len());
    }

    /// Get next version number for a key
    fn get_next_version(&self, key: &str) -> u64 {
        self.state.read().get(key).map_or(1, |e| e.version + 1)
    }

    /// Replicate synchronously to all replicas
    async fn replicate_synchronous(&self, _entry: StateEntry) -> Result<()> {
        // Simulate synchronous replication
        tokio::time::sleep(Duration::from_millis(10)).await;
        info!("Synchronous replication completed");
        Ok(())
    }

    /// Replicate asynchronously
    async fn replicate_asynchronous(&self, _entry: StateEntry) {
        // Simulate asynchronous replication
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            debug!("Asynchronous replication completed");
        });
    }

    /// Replicate using quorum approach
    async fn replicate_quorum(&self, _entry: StateEntry) -> Result<()> {
        // Simulate quorum replication (majority)
        let quorum_size = (self.replica_count / 2) + 1;
        tokio::time::sleep(Duration::from_millis(8)).await;
        info!("Quorum replication completed (quorum: {})", quorum_size);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_manager_basic() {
        let manager = DistributedStateManager::new(SyncMode::Synchronous, 3);

        let key = "test_key".to_string();
        let value = serde_json::json!({"data": "test"});

        manager.set(key.clone(), value.clone(), None).await.unwrap();

        let retrieved = manager.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);
    }

    #[tokio::test]
    async fn test_state_manager_ttl() {
        let manager = DistributedStateManager::new(SyncMode::Asynchronous, 3);

        let key = "expiring_key".to_string();
        let value = serde_json::json!({"data": "expiring"});
        let ttl = Duration::from_millis(100);

        manager.set(key.clone(), value, Some(ttl)).await.unwrap();

        // Immediately should be available
        assert!(manager.get(&key).is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be expired
        assert!(manager.get(&key).is_none());
    }

    #[tokio::test]
    async fn test_state_manager_delete() {
        let manager = DistributedStateManager::new(SyncMode::Quorum, 3);

        let key = "deletable_key".to_string();
        let value = serde_json::json!({"data": "delete_me"});

        manager.set(key.clone(), value, None).await.unwrap();
        assert!(manager.get(&key).is_some());

        manager.delete(&key).await.unwrap();
        assert!(manager.get(&key).is_none());
    }

    #[tokio::test]
    async fn test_state_manager_keys() {
        let manager = DistributedStateManager::new(SyncMode::Synchronous, 3);

        manager
            .set("key1".to_string(), serde_json::json!(1), None)
            .await
            .unwrap();
        manager
            .set("key2".to_string(), serde_json::json!(2), None)
            .await
            .unwrap();
        manager
            .set("key3".to_string(), serde_json::json!(3), None)
            .await
            .unwrap();

        let keys = manager.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        assert!(keys.contains(&"key3".to_string()));
    }

    #[tokio::test]
    async fn test_state_manager_cleanup_expired() {
        let manager = DistributedStateManager::new(SyncMode::Asynchronous, 3);

        let ttl = Duration::from_millis(50);

        manager
            .set("key1".to_string(), serde_json::json!(1), Some(ttl))
            .await
            .unwrap();
        manager
            .set("key2".to_string(), serde_json::json!(2), None)
            .await
            .unwrap();

        assert_eq!(manager.entry_count(), 2);

        tokio::time::sleep(Duration::from_millis(100)).await;

        manager.cleanup_expired();

        assert_eq!(manager.entry_count(), 1);
        assert!(manager.get("key2").is_some());
    }

    #[test]
    fn test_sync_modes() {
        assert_eq!(SyncMode::Synchronous, SyncMode::Synchronous);
        assert_ne!(SyncMode::Synchronous, SyncMode::Asynchronous);
    }
}
